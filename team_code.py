#!/usr/bin/env python

# 简化版team_code.py，专注于SECNN模型

import os
import numpy as np
import torch
import wfdb
from tqdm import tqdm

# 导入必要的挑战赛辅助函数
from helper_code import *

# 导入我们自己的模型和训练代码
# from train_bceloss_secnn import train_secnn_model
# from model_SECNN_BCEloss import SECNN


from train_celoss_secnn import train_secnn_model
from model_SECNN_CEloss import SECNN


import config


# 训练模型函数
def train_model(data_folder, model_folder, verbose):
    if verbose:
        print('寻找挑战赛数据...')

    # 检查数据文件夹是否存在
    if not os.path.exists(data_folder):
        raise FileNotFoundError(f'数据文件夹未找到: {data_folder}')

    # 训练SECNN模型
    if verbose:
        print('训练SECNN模型...')

    # 调用简化版SECNN训练函数
    model_dict = train_secnn_model(
        data_folder=data_folder,
        model_folder=model_folder,
        epochs=config.epochs,  # 可以根据需要调整
        verbose=verbose
    )

    if verbose:
        print('模型训练完成。')


# 加载训练好的模型
def load_model(model_folder, verbose):
    if verbose:
        print('加载模型...')

    # 检查路径是否存在
    if not os.path.exists(model_folder):
        raise FileNotFoundError(f'模型文件夹未找到: {model_folder}')

    # 检查是否是具体的模型目录路径
    if os.path.basename(model_folder).startswith('chagas_model_'):
        # 直接使用提供的具体模型目录
        model_path = os.path.join(model_folder, 'secnn_model.pt')
        checkpoint_path = os.path.join(model_folder, 'best_model.pth')
    else:
        # 原有逻辑：查找最新的模型文件夹
        model_dirs = [d for d in os.listdir(model_folder) if
                      d.startswith('chagas_model_') and os.path.isdir(os.path.join(model_folder, d))]
        if not model_dirs:
            raise FileNotFoundError(f'在 {model_folder} 中未找到模型目录')

        # 按时间戳排序，选择最新的
        latest_model_dir = sorted(model_dirs)[-1]
        model_path = os.path.join(model_folder, latest_model_dir, 'secnn_model.pt')
        checkpoint_path = os.path.join(model_folder, latest_model_dir, 'best_model.pth')

    # 加载模型
    if os.path.exists(model_path):
        # 直接加载保存的模型
        model = torch.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    elif os.path.exists(checkpoint_path):
        # 从检查点加载
        checkpoint = torch.load(checkpoint_path,
                                map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        model = SECNN(input_length=4096, num_classes=config.num_classes)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        raise FileNotFoundError(f'模型文件未找到: {model_path} 或 {checkpoint_path}')

    # 设置设备
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 设置为评估模式
    model.eval()

    if verbose:
        print('模型加载成功。')

    return model


# 运行模型进行预测
def run_model(record, model, verbose):
    # if verbose:
        # print(f'对记录 {record} 运行模型')

    # 提取特征和元数据
    signal, metadata = extract_features_for_secnn(record)

    if not isinstance(signal, torch.Tensor):
        signal = torch.tensor(signal, dtype=torch.float32)
    else:
        signal = signal.float()  # 确保是float类型

    if not isinstance(metadata, torch.Tensor):
        metadata = torch.tensor(metadata, dtype=torch.float32)
    else:
        metadata = metadata.float()

    # 预测
    binary_output, probability_output = predict_with_secnn(model, signal, metadata)

    # 转换输出格式以符合挑战赛要求
    if isinstance(binary_output, np.ndarray) and len(binary_output) == 1:
        binary_output = binary_output[0]
    if isinstance(probability_output, np.ndarray) and len(probability_output) == 1:
        probability_output = probability_output[0]

    # 确保二元输出是0或1
    binary_output = int(binary_output)

    # if verbose:
    #     print(f'预测: {binary_output}, 概率: {probability_output:.4f}')

    return binary_output, probability_output

def predict_with_secnn(model, signal, metadata):
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    # 确保信号是PyTorch张量，形状为[1, 12, seq_len]
    if not isinstance(signal, torch.Tensor):
        signal = torch.tensor(signal, dtype=torch.float32)

    if signal.dim() == 2:  # [12, seq_len]
        signal = signal.unsqueeze(0)  # [1, 12, seq_len]

    # 确保元数据也是PyTorch张量，并增加批次维度
    if not isinstance(metadata, torch.Tensor):
        metadata = torch.tensor(metadata, dtype=torch.float32)

    if metadata.dim() == 1:  # [2]
        metadata = metadata.unsqueeze(0)  # [1, 2]

    # 打印信号形状进行调试
    # print(f"输入信号形状: {signal.shape}")

    # 将数据移到正确的设备
    signal = signal.to(device)
    metadata = metadata.to(device)

    with torch.no_grad():
        # 尝试直接进行推理前打印模型结构
        # print(f"模型信息: {model.__class__.__name__}")

        logits, _ = model(signal, metadata)  # 不使用元数据进行预测

        # 对于CrossEntropyLoss，使用softmax获取概率
        probs = torch.softmax(logits, dim=1)
        probability = probs[:, 1].cpu().numpy()[0]  # 获取正类的概率
        predicted = logits.argmax(dim=1).item()

    return predicted, probability


# 为SECNN模型提取特征
def extract_features_for_secnn(record):
    try:
        signal, fields = load_signals(record)
        # print(f"成功加载记录 {record} 信号，形状: {signal.shape}")

        # 同时加载元数据
        header = load_header(record)
        age = get_age(header)
        if np.isnan(age):
            age = 50.0  # 使用默认值

        sex = get_sex(header)
        sex_encoded = 1.0 if sex == 'Male' else 0.0 if sex == 'Female' else 0.5

        # 将元数据组合成一个数组
        metadata = np.array([age, sex_encoded], dtype=np.float32)


    except Exception as e:
        try:
            record_data = wfdb.rdrecord(record)
            signal = record_data.p_signal
            # 如果这种方式加载的话，可能没有元数据，使用默认值
            metadata = np.array([50.0, 0.5], dtype=np.float32)  # 默认年龄和性别中性值
        except Exception as e2:
            print(f"加载记录 {record} 时出错: {e2}")
            signal = np.zeros((4096, 12), dtype=np.float32)
            metadata = np.array([50.0, 0.5], dtype=np.float32)  # 默认值

    # 打印原始信号形状和类型
    # print(f"原始信号形状: {signal.shape}, 类型: {signal.dtype}")
    # 处理信号
    if len(signal.shape) == 1:
        signal = np.expand_dims(signal, axis=1)  # 如果是一维数据，扩展维度以匹配12导联格式
    signal_12leads = signal

    # 处理信号长度
    max_length = 4096
    if signal_12leads.shape[0] > max_length:
        start = (signal_12leads.shape[0] - max_length) // 2
        signal_12leads = signal_12leads[start:start + max_length, :]
    elif signal_12leads.shape[0] < max_length:
        padding = np.zeros((max_length - signal_12leads.shape[0], 12), dtype=signal_12leads.dtype)
        signal_12leads = np.concatenate([signal_12leads, padding], axis=0)

    # 对每个导联进行标准化
    for lead in range(12):
        lead_signal = signal_12leads[:, lead]
        mean = np.mean(lead_signal)
        std = np.std(lead_signal)
        if std < 1e-6:
            std = 1.0  # 避免除以零
        signal_12leads[:, lead] = (lead_signal - mean) / std

    # 处理NaN和异常值
    signal_12leads = np.nan_to_num(signal_12leads, nan=0.0, posinf=0.0, neginf=0.0)
    signal_12leads = np.clip(signal_12leads, -5.0, 5.0)

    # 转置为[12, seq_len]形状，与PyTorch卷积层期望的输入兼容
    return signal_12leads.transpose(1, 0), metadata  # 返回形状为 [12, seq_len] 的信号和元数据