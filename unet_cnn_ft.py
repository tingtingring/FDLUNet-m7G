import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda", 0)

class ConvBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock1D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class ConvTransformerBlock(nn.Module):
    def __init__(self, in_channels, out_channels, nhead=8, num_layers=6):
        super().__init__()
        self.conv = ConvBlock1D(in_channels, out_channels)

        self.layernorm = nn.LayerNorm(out_channels)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=out_channels,
            nhead=nhead,
            dim_feedforward=out_channels*4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)


    def forward(self, x):
        # x: (B, C_in, L)
        x = self.conv(x)                # → (B, C_out, L)
        x = x.permute(0, 2, 1)          # → (B, L, C_out)

        x = self.layernorm(x)
        x = self.transformer(x)         # → (B, L, C_out)

        x = x.permute(0, 2, 1)          # → (B, C_out, L)
        return x


class UNet1D(nn.Module):
    def __init__(self, in_channels=64, base_filters=64):
        super(UNet1D, self).__init__()
        filters = [base_filters, base_filters*2, base_filters*4]

        # Encoder
        self.enc1 = ConvTransformerBlock(in_channels, filters[0])
        self.pool1 = PaddedMaxPool1d(2)

        self.enc2 = ConvTransformerBlock(filters[0], filters[1])
        self.pool2 = PaddedMaxPool1d(2)

        self.enc3 = ConvTransformerBlock(filters[1], filters[2])

        # Decoder
        self.up2 = CropConvTranspose1d(filters[2], filters[1], kernel_size=2, stride=2)
        self.dec2 = ConvBlock1D(filters[2], filters[1])

        self.up1 = CropConvTranspose1d(filters[1], filters[0], kernel_size=2, stride=2)
        self.dec1 = ConvBlock1D(filters[1], filters[0])

        # Final conv
        self.final_conv = nn.Conv1d(filters[0], filters[0], kernel_size=1)

    def forward(self, x):
        # Encoder path
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        # Decoder path
        d2 = self.up2(e3, e2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2, e1)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        # 输出最终的特征图
        features = self.final_conv(d1)  # shape: (B, C, L)

        return features  # 不做分类

class PaddedMaxPool1d(nn.Module):
    """
    自动在右边补 1，使 MaxPool1d 输出为 ceil(L/2)
    """
    def __init__(self, kernel_size=2, stride=2):
        super().__init__()
        self.pool = nn.MaxPool1d(kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        # 如果长度是奇数，就右边 padding 一个值
        if x.size(-1) % 2 != 0:
            x = F.pad(x, (0, 1))  # pad: (left, right)
        return self.pool(x)

class CropConvTranspose1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0, output_padding=0, bias=True):
        super().__init__()
        self.trans_conv = nn.ConvTranspose1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            bias=bias
        )

    @staticmethod
    def center_crop_to_match(source, target):
        diff = source.size(-1) - target.size(-1)
        if diff < 0:
            raise ValueError(f"Cannot crop: source length {source.size(-1)} smaller than target length {target.size(-1)}")
        start = diff // 2
        end = start + target.size(-1)
        return source[:, :, start:end]

    def forward(self, x, skip):
        """
        x: decoder输入，shape (B, C_in, L_in)
        skip: encoder对应skip连接，shape (B, C_skip, L_skip)

        返回：
        upsampled: 反卷积结果，shape (B, C_out, L_up)
        skip_cropped: 裁剪后与 upsampled 长度匹配的 skip，shape (B, C_skip, L_up)
        """
        upsampled = self.trans_conv(x)  # (B, C_out, L_up)
        skip_cropped = self.center_crop_to_match(upsampled,skip)
        return skip_cropped

class FTNET(nn.Module):
    def __init__(self, params):
        super(FTNET, self).__init__()
        self.params = params
        self.seq_len = params['seq_len']

        # CNN预处理层
        self.pre_cnn = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
        )

        # 三个 U-Net 分支
        self.unet_low = UNet1D(in_channels=64)
        self.unet_high = UNet1D(in_channels=64)
        self.unet_raw = UNet1D(in_channels=64)

        # 频域分支卷积
        self.conv_low = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.conv_high = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.conv_raw = nn.Conv1d(64, 64, kernel_size=3, padding=1)

        # 特征融合层
        self.fusion_fc = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Sigmoid()
        )

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 编码序列
        x = self.encode_sequence_1mer(x, self.seq_len)
        x = torch.tensor(x, dtype=torch.long).to(device)

        # One-hot
        sequence = F.one_hot(x, num_classes=4).permute(0, 2, 1).float()

        # 初步卷积
        features = self.pre_cnn(sequence)  #

        # ============ FFT分解 ============
        fft_features = torch.fft.fft(features, dim=-1)
        freq_len = fft_features.size(-1)
        half = freq_len // 2

        # 低频掩码
        low_mask = torch.zeros_like(fft_features)
        low_mask[..., :half//2] = 1
        low_mask[..., -half//2:] = 1
        high_mask = 1 - low_mask

        # 掩码分离
        fft_low = fft_features * low_mask
        fft_high = fft_features * high_mask

        feat_low = torch.fft.ifft(fft_low, dim=-1).real
        feat_high = torch.fft.ifft(fft_high, dim=-1).real

        # ============ 三分支提取 ============
        out_low = self.unet_low(feat_low)  #64,64,201
        out_high = self.unet_high(feat_high)
        out_raw = self.unet_raw(features)

        # 卷积调整通道
        out_low = self.conv_low(out_low)
        out_high = self.conv_high(out_high)
        out_raw = self.conv_raw(out_raw)

        # 拼接频域与原始特征
        f_low_mean = torch.mean(out_low, dim=-1)
        f_high_mean = torch.mean(out_high, dim=-1)
        f_raw_mean = torch.mean(out_raw, dim=-1)

        fused =  f_low_mean + f_high_mean + f_raw_mean # [B,64]

        # 融合分类
        out_mask = self.fusion_fc(fused)

        # 残差式乘回原特征均值（可选）
        out_feature = out_raw + out_raw * out_mask.unsqueeze(-1)

        # ===== 变换维度 + 取中间位置 =====
        out_feature = out_feature.permute(0, 2, 1)  # [B,L,64]
        center_idx = out_feature.size(1) // 2
        center_feature = out_feature[:, center_idx, :]  # [B,64]

        # ===== 分类 =====
        pred = self.classifier(center_feature)  # [B,1]

        return pred.squeeze(-1)


    def encode_sequence_1mer(self, sequences, max_seq=201):
        k = 1
        overlap = False
        all_kmer = [''.join(p) for p in itertools.product(['A', 'T', 'C', 'G', '-'], repeat=k)]
        kmer_dict = {all_kmer[i]: i for i in range(len(all_kmer))}

        encoded_sequences = []
        max_length = max_seq - k + 1 if overlap else max_seq // k

        for seq in sequences:
            encoded_seq = []
            start_site = len(seq) // 2 - max_length // 2
            for i in range(start_site, start_site + max_length, k):
                encoded_seq.append(kmer_dict.get(seq[i:i + k], 0))
            encoded_sequences.append(encoded_seq + [0] * (max_length - len(encoded_seq)))

        return np.array(encoded_sequences)

