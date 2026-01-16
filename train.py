import torch
import torch.nn as nn
import torch.optim as optim

import torch.utils.data as Data
import torch.nn.functional as F
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import itertools
import math
import numpy as np
import os
import random

import time

tqdm.pandas(ascii=True)
import os
from sklearn.metrics import auc, roc_curve, precision_recall_curve, average_precision_score
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, TensorDataset
from termcolor import colored
import datetime

from torch.utils.tensorboard import SummaryWriter
from unet_cnn_ft import FTNET

import copy

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device = torch.device("cuda", 0)



def read_fasta(file):
    seq = []
    label = []
    with open(file) as fasta:
        for line in fasta:
            line = line.replace('\n', '')
            if line.startswith('>'):
                # label.append(int(line[-1]))
                if 'neg' in line:
                    label.append(0)
                else:
                    label.append(1)
            else:
                seq.append(line.replace('U', 'T'))

    return seq, label

def read_fasta_pretrain(dir_path, prefix='1'):     #data1-data10分别对应于Am, Cm, Gm, Um, m1A, m5C, m5U, m6A, m6Am, Ψ
    seq = []
    label = []

    for target_label in [0, 1]:
        file_path = os.path.join(dir_path, f'{prefix}-{target_label}.fasta')
        if not os.path.exists(file_path):
            print(f"文件不存在: {file_path}")
            continue

        with open(file_path) as fasta:
            current_seq = ""
            for line in fasta:
                line = line.strip()
                if line.startswith('>'):
                    if current_seq:
                        seq.append(current_seq)
                        label.append(target_label)
                        current_seq = ""
                else:
                    current_seq += line.replace('U', 'T')
            if current_seq:
                seq.append(current_seq)
                label.append(target_label)

    return seq, label

def read_fasta_from_m5C(file):
    seq = []
    label = []
    with open(file) as fasta:
        for line in fasta:
            line = line.strip()
            if line.startswith('>'):
                # 例如：>chr5|113026500|0|train
                parts = line[1:].split('|')  # 去掉 '>' 后分割
                try:
                    lab = int(parts[1])  # 获取标签
                except (IndexError, ValueError):
                    print(f"Header 格式错误: {line}")
                    lab = 0  # 默认设为负类
                label.append(lab)
            else:
                seq.append(line.replace('U', 'T'))  # 替换 RNA 的 U 为 T

    return seq, label

def read_fasta_from_AtoI(file):
    seq = []
    label = []
    with open(file) as fasta:
        for line in fasta:
            line = line.strip()
            if line.startswith('>'):
                # 格式为：>chr19:34401574|1|train
                try:
                    parts = line[1:].split('|')  # 去掉 >，然后分割
                    lab = int(parts[1])  # 第二段是标签
                    label.append(lab)
                except (IndexError, ValueError):
                    print(f"Header 格式错误: {line}")
                    label.append(0)  # 默认负类
            else:
                seq.append(line.replace('U', 'T'))  # 替换 U 为 T（如需RNA→DNA）

    return seq, label



class MyDataSet(Dataset):
    def __init__(self, data, label):
        self.data = data  # ['今天天气很好', 'xxxx', ....]
        self.label = label  # [1, 0, 2]

    def __getitem__(self, idx):
        # 获取原始文本和标签
        text = self.data[idx]  # str
        label = self.label[idx]

        return  text, label

    def __len__(self):
        return len(self.data)


def encode_sequence_1mer(sequences, max_seq):
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

        # 补齐到 max_length 长度
        encoded_sequences.append(encoded_seq + [0] * (max_length - len(encoded_seq)))

    return np.array(encoded_sequences)

def evaluation_method(params):

    # train_x, train_y = read_fasta_from_AtoI('data/m1A/train_65.fasta')
    # valid_x, valid_y = read_fasta_from_AtoI('data/m1A/val_65.fasta')
    # test_x, test_y = read_fasta_from_AtoI('data/m1A/test_65.fasta')
    #
    # index = params['index']
    # train_x, train_y = read_fasta_pretrain('data_pretrain/train',index)
    # valid_x, valid_y = read_fasta_pretrain('data_pretrain/valid',index)
    # test_x, test_y = read_fasta_pretrain('data_pretrain/test',index)

    train_x, train_y = read_fasta_from_AtoI('data_new/m7G/train_ref.fasta')
    valid_x, valid_y = read_fasta_from_AtoI('data_new/m7G/val_ref.fasta')
    test_x, test_y = read_fasta_from_AtoI('data_new/m7G/test_ref.fasta')

    # train_x, train_y = read_fasta('data/origin/train.fasta')
    # valid_x, valid_y = read_fasta('data/origin/valid.fasta')
    # test_x, test_y = read_fasta('data/origin/test.fasta')

    # train_x, train_y = read_fasta_from_AtoI('data/AtoI_advanced/train.fasta')
    # valid_x, valid_y = read_fasta_from_AtoI('data/AtoI_advanced/val.fasta')
    # test_x, test_y = read_fasta_from_AtoI('data/AtoI_advanced/test.fasta')

    # train_x, train_y = read_fasta_from_m5C('data/m5C/train.fasta')
    # valid_x, valid_y = read_fasta_from_m5C('data/m5C/val.fasta')
    # test_x, test_y = read_fasta_from_m5C('data/m5C/test.fasta')

    train_x, train_y = list(train_x), list(train_y)
    valid_x, valid_y = list(valid_x),list(valid_y)
    test_x, test_y = list(test_x),list(test_y)

    train_dataset = MyDataSet(train_x, train_y)
    valid_dataset = MyDataSet(valid_x, valid_y)
    test_dataset = MyDataSet(test_x, test_y)

    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=params['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False)

    train_model(train_loader, valid_loader, test_loader, params)

def to_log(log, params, start_time):
    # 创建保存路径
    seq_len = params['seq_len']
    seed = params['seed']
    log_dir = f"results/seq_len{seq_len}"
    os.makedirs(log_dir, exist_ok=True)  # 如果文件夹不存在，创建文件夹

    # 日志文件名包含 seed、seq_len 和开始时间
    log_path = f"{log_dir}/train_diff_len_seed{seed}_seq_len{seq_len}_{start_time}.log"

    # 写入日志
    with open(log_path, "a+") as f:
        f.write(log + '\n')

def train_model(train_loader, valid_loader, test_loader, params):
    # 获取训练开始的时间戳
    start_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    model = FTNET(params).to(device)

    # Optimizer and loss
    opt = optim.Adam(model.parameters(), lr=params['lr'], betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
    criterion_BCE = nn.BCELoss()
    best_acc = 0
    patience = params['patience']
    now_epoch = 0

    best_model = None
    for epoch in range(params['epoch']):
        model.train()
        loss_ls = []
        t0 = time.time()
        for seq, label in tqdm(train_loader):
            seq, label = seq, label.to(device)

            output = model(seq)

            loss = criterion_BCE(output, label.float())

            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_ls.append(loss.item())

        model.eval()
        with torch.no_grad():
            train_performance, train_roc_data, train_prc_data = evaluate(train_loader, model)
            valid_performance, valid_roc_data, valid_prc_data = evaluate(valid_loader, model)

        results = f"\nepoch: {epoch + 1}, loss: {np.mean(loss_ls):.5f}\n"
        results += f'Train: {train_performance[0]:.4f}, time: {time.time() - t0:.2f}'
        results += '\n' + '=' * 16 + ' Valid Performance. Epoch[{}] '.format(epoch + 1) + '=' * 16 \
                   + '\n[ACC, \tBACC, \tSE,\t\tSP,\t\tMCC,\tAUC]\n' + '{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f}'.format(
            valid_performance[0], valid_performance[1], valid_performance[2], valid_performance[3],
            valid_performance[4], valid_performance[5]) + '\n' + '=' * 60
        valid_acc = valid_performance[0]  # test_performance: [ACC, Sensitivity, Specificity, AUC, MCC]
        print(results)
        to_log(results, params, start_time)


        if valid_acc > best_acc:
            best_acc = valid_acc
            now_epoch = 0
            best_model = copy.deepcopy(model)
            to_log('here occur best!\n', params, start_time)

            # 创建保存路径
            seq_len = params['seq_len']
            seed = params['seed']
            save_dir = f"save/seq_len{seq_len}"
            os.makedirs(save_dir, exist_ok=True)  # 如果文件夹不存在，创建文件夹

            # 文件名包含 seed、开始时间和 best_acc
            save_path = f"{save_dir}/seed{seed}_{start_time}_acc{best_acc:.4f}.pth"

            # 保存模型
            checkpoint = {
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
                'epoch': epoch + 1,
                'params': params
            }
            torch.save(checkpoint, save_path)

            print(f"Checkpoint saved to {save_path}")

        else:
            now_epoch += 1
            print('now early stop target = ', now_epoch)
        test_performance, test_roc_data, test_prc_data = evaluate(test_loader, model)
        test_results = '\n' + '=' * 16 + colored(' Test Performance. Epoch[{}] ', 'red').format(
            epoch + 1) + '=' * 16 \
                       + '\n[ACC,\tBACC, \tSE,\t\tSP,\t\tMCC,\tAUC]\n' + '{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f}'.format(
            test_performance[0], test_performance[1], test_performance[2], test_performance[3],
            test_performance[4], test_performance[5]) + '\n' + '=' * 60
        print(test_results)
        to_log(test_results, params, start_time)

        if now_epoch > patience:
            print('early stop!!!')
            best_performance, best_roc_data, best_prc_data = evaluate(test_loader, best_model)
            best_results = '\n' + '=' * 16 + colored(' Test Performance. Early Stop ', 'red').format(
                epoch + 1) + '=' * 16 \
                           + '\n[ACC,\tBACC, \tSE,\t\tSP,\t\tMCC,\tAUC]\n' + '{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f}'.format(
                best_performance[0], best_performance[1], best_performance[2], best_performance[3],
                best_performance[4], best_performance[5]) + '\n' + '=' * 60
            print(best_results)
            to_log(best_results, params, start_time)
            break


def caculate_metric(pred_prob, label_pred, label_real):
    test_num = len(label_real)
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for index in range(test_num):
        if label_real[index] == 1:
            if label_real[index] == label_pred[index]:
                tp = tp + 1
            else:
                fn = fn + 1
        else:
            if label_real[index] == label_pred[index]:
                tn = tn + 1
            else:
                fp = fp + 1

    # Accuracy
    ACC = float(tp + tn) / test_num

    # Sensitivity
    if tp + fn == 0:
        Recall = Sensitivity = 0
    else:
        Recall = Sensitivity = float(tp) / (tp + fn)

    # Specificity
    if tn + fp == 0:
        Specificity = 0
    else:
        Specificity = float(tn) / (tn + fp)

    # MCC
    if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) == 0:
        MCC = 0
    else:
        MCC = float(tp * tn - fp * fn) / np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))

    # ROC and AUC
    FPR, TPR, thresholds = roc_curve(label_real, pred_prob, pos_label=1)

    AUC = auc(FPR, TPR)

    # PRC and AP
    precision, recall, thresholds = precision_recall_curve(label_real, pred_prob, pos_label=1)
    AP = average_precision_score(label_real, pred_prob, average='macro', pos_label=1, sample_weight=None)

    if (tp + fp) == 0:
        PRE = 0
    else:
        PRE = float(tp) / (tp + fp)

    BACC = 0.5 * Sensitivity + 0.5 * Specificity

    performance = [ACC, BACC, Sensitivity, Specificity, MCC, AUC]
    roc_data = [FPR, TPR, AUC]
    prc_data = [recall, precision, AP]
    return performance, roc_data, prc_data

def evaluate(data_iter, net):
    net.eval()
    pred_prob_main = []
    label_real = []

    with torch.no_grad():
        for data, labels in data_iter:
            data, labels = data, labels.to(device)
            outputs_main = net(data)

            # 主输出的评估数据
            if outputs_main.dim() == 2 and outputs_main.shape[1] == 1:
                pred_prob_main.extend(outputs_main.squeeze(-1).cpu().numpy().tolist())
            else:
                pred_prob_main.extend(outputs_main.cpu().numpy().tolist())


            label_real.extend(labels.cpu().numpy().tolist())

    # 主输出的性能指标
    performance_main, roc_data_main, prc_data_main = caculate_metric(
        pred_prob_main, (np.array(pred_prob_main) > 0.5).astype(int).tolist(), label_real
    )


    return performance_main, roc_data_main, prc_data_main

def main():

    params = {
        'lr': 0.0001,
        'batch_size': 64,
        'epoch': 100,
        'seq_len': 201,
        'saved_model_name': 'diff_len_',
        'seed': 17,
        'patience': 10,
        'index':10
    }
    seed = params['seed']
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    evaluation_method(params)

if __name__ == '__main__':
    main()