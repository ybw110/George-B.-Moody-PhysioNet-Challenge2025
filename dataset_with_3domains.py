#!/usr/bin/env python

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import wfdb
from collections import defaultdict
import random
from helper_code import load_header, load_signals, load_label, get_age, get_sex
from data_aug import *


class MultiSourceChagasECGDataset(Dataset):

    def __init__(self, records, folder_map, max_length=4096, transform=None, balance=False, source_weights=None):

        self.records = records
        self.folder_map = folder_map
        self.max_length = max_length
        self.transform = transform
        self.source_weights = source_weights or {}  # 添加源权重字典

        # 预定义增强管道
        if transform is None:
            self.transform = Compose([
                Reshape(),
                Retype()
            ])

        # 预先加载标签和人口统计学信息
        self.labels = []  # 存储每个记录的Chagas标签（0或1）
        self.ages = []  # 存储每个记录的年龄
        self.sexes = []  # 存储每个记录的性别（编码为0、0.5或1）
        self.dlabels = []  # 存储每个记录的领域标签，默认为0
        self.valid_indices = []  # 存储有效记录的索引（有标签的记录）
        self.source_labels = []  # 记录每个样本来自哪个数据源
        self.sample_weights = []  # 添加样本权重
        self._current_idx = 0  # 添加当前索引跟踪

        pos_count = 0  # 初始化正样本和负样本计数器，用于计算类别权重和报告类别分布
        neg_count = 0
        source_stats = defaultdict(int)  # 添加源统计

        for i, record in enumerate(records):  # 遍历所有记录，构建完整文件路径，并使用try-except块处理潜在的错误。
            # 获取原始记录ID和文件夹
            folder = self.folder_map[record]['folder']
            original_id = self.folder_map[record]['original_id']
            source = self.folder_map[record].get('source', 'unknown')  # 获取数据源
            record_path = os.path.join(folder, original_id)

            try:
                header = load_header(record_path)  # 加载标签和元数据,xxxxx.hea,加载WFDB头文件，包含元数据如年龄、性别和Chagas标签

                try:  # 获取标签,应该就是True,False

                    label = load_label(record_path)  # 使用 load_label 函数从头文件中提取标签
                    self.labels.append(float(label))  # 将标签转换为浮点数并添加到列表
                    self.dlabels.append(0.0)  # 为每个有效记录添加领域标签，默认为0
                    self.source_labels.append(source)  # 记录数据源
                    source_stats[source] += 1  # 统计数据源分布

                    weight = self.source_weights.get(source, 1.0)  # 根据源获取权重
                    if label and source == 'SaMi-Trop':  # 只增加SaMi-Trop的阳性样本权重
                        weight *= 2.0
                    self.sample_weights.append(weight)

                    if label:  # 统计正负样本数量
                        pos_count += 1
                    else:
                        neg_count += 1

                except Exception:  # 没有标签的记录跳过
                    continue

                age = get_age(header)  # 获取年龄
                if np.isnan(age):
                    age = 50.0  # 使用默认值
                self.ages.append(age)

                sex = get_sex(header)  # 获取性别
                sex_encoded = 1.0 if sex == 'Male' else 0.0 if sex == 'Female' else 0.5
                self.sexes.append(sex_encoded)

                self.valid_indices.append(i)  # 将当前记录的索引添加到valid_indices列表，表示这是一个有效的记录（有标签）

            except Exception as e:
                print(f"Error processing record {record}: {e}")

        # 转换为numpy数组
        self.labels = np.array(self.labels)
        self.ages = np.array(self.ages)
        self.sexes = np.array(self.sexes)
        self.dlabels = np.array(self.dlabels)
        self.source_labels = np.array(self.source_labels)
        self.sample_weights = np.array(self.sample_weights)

        # 计算类别权重
        self.pos_count = pos_count
        self.neg_count = neg_count
        self.pos_weight = self.neg_count / max(1, self.pos_count)

        print(f"Dataset statistics:")
        print(f"  Total valid records: {len(self.valid_indices)}")
        print(f"  Positive samples: {self.pos_count} ({self.pos_count / len(self.valid_indices) * 100:.2f}%)")
        print(f"  Negative samples: {self.neg_count} ({self.neg_count / len(self.valid_indices) * 100:.2f}%)")
        print(f"  Source distribution: {dict(source_stats)}")

        # 如果 balance=True，使用更复杂的平衡策略
        if balance:
            self.balanced_indices = self._balance_classes_by_source()
            print(f"Balanced dataset: {len(self.balanced_indices)} records")
        else:
            self.balanced_indices = self.valid_indices

    # 将此方法添加到 dataset_with_latent_domain.py 中的 MultiSourceChagasECGDataset 类

    def set_labels_by_index(self, pred_labels, indices, label_type='pdlabel'):
        try:
            # 确保输入类型正确
            pred_labels = np.array(pred_labels)
            indices = np.array(indices)

            if label_type == 'pdlabel':
                # 为指定索引更新域标签
                for idx, label in zip(indices, pred_labels):
                    idx_int = int(idx)  # 将 float32 转换为 int
                    real_idx = self.balanced_indices[idx] if idx < len(self.balanced_indices) else idx
                    if 0 <= real_idx < len(self.dlabels):
                        self.dlabels[real_idx] = int(label)
                    else:
                        print(f"警告：索引 {real_idx} 超出范围")
            else:
                raise ValueError(f"不支持的标签类型: {label_type}")

            print(f"成功更新了 {len(indices)} 个标签")
        except Exception as e:
            print(f"set_labels_by_index 出错: {e}")
            raise

    def _balance_classes_by_source(self):
        # 按数据源和标签分类样本
        source_pos_indices = defaultdict(list)
        source_neg_indices = defaultdict(list)

        for i in self.valid_indices:
            source = self.source_labels[i]
            label = self.labels[i]

            if label > 0:
                source_pos_indices[source].append(i)
            else:
                source_neg_indices[source].append(i)

        # 保留所有SaMi-Trop样本
        samitrop_samples = source_pos_indices['SaMi-Trop'] + source_neg_indices['SaMi-Trop']
        samitrop_count = len(samitrop_samples)
        print(f"SaMi-Trop样本总数: {samitrop_count}")
        print(f"  - 阳性: {len(source_pos_indices['SaMi-Trop'])}")
        print(f"  - 阴性: {len(source_neg_indices['SaMi-Trop'])}")

        # PTB-XL抽样与SaMi-Trop数量相同
        ptbxl_target = samitrop_count* 2
        ptbxl_samples = []
        if ptbxl_target > 0:
            # 优先保留PTB-XL中的阳性样本(如果有)
            ptbxl_pos = source_pos_indices['PTB-XL']
            ptbxl_samples.extend(ptbxl_pos)

            # 剩余配额用阴性样本填充
            remaining = ptbxl_target - len(ptbxl_pos)
            if remaining > 0 and source_neg_indices['PTB-XL']:
                ptbxl_neg_samples = random.sample(
                    source_neg_indices['PTB-XL'],
                    min(remaining, len(source_neg_indices['PTB-XL']))
                )
                ptbxl_samples.extend(ptbxl_neg_samples)

        print(f"PTB-XL抽样数: {len(ptbxl_samples)}")

        # CODE-15%抽样为SaMi-Trop和PTB-XL总量的3倍
        code15_target = (samitrop_count + len(ptbxl_samples)) * 3
        code15_samples = []

        # 为CODE-15%保持阳性/阴性样本比例
        code15_pos = source_pos_indices['CODE-15%']
        code15_neg = source_neg_indices['CODE-15%']
        print(f"CODE-15%原始数据: 阳性 {len(code15_pos)}, 阴性 {len(code15_neg)}, 总计 {len(code15_pos) + len(code15_neg)}")

        # 优先保留所有阳性样本
        code15_samples.extend(code15_pos)
        print(f"已保留所有CODE-15%阳性样本: {len(code15_pos)}")
        # 计算需要补充的阴性样本数量
        neg_needed = code15_target - len(code15_pos)
        print(f"需要补充阴性样本: {neg_needed}")

        # 如果阴性样本数量足够，随机抽样补充
        if len(code15_neg) > 0:
            neg_to_add = min(neg_needed, len(code15_neg))
            code15_neg_samples = random.sample(code15_neg, neg_to_add)
            code15_samples.extend(code15_neg_samples)
            print(f"已补充阴性样本: {len(code15_neg_samples)}")

        # 如果阴性样本数量不够，可以考虑重复使用阳性样本(过采样)来达到目标数量
        if len(code15_samples) < code15_target and len(code15_pos) > 0:
            additional_needed = code15_target - len(code15_samples)
            print(f"阴性样本不足，需要进一步补充: {additional_needed}")
            # 使用有放回抽样(过采样)来补充阳性样本
            additional_pos = random.choices(code15_pos, k=additional_needed)
            code15_samples.extend(additional_pos)
            print(f"已通过过采样补充阳性样本: {additional_needed}")

        print(f"CODE-15%最终抽样数: {len(code15_samples)}")
        print(f"  - 阳性: {sum(1 for i in code15_samples if self.labels[i] > 0)}")
        print(f"  - 阴性: {sum(1 for i in code15_samples if self.labels[i] == 0)}")
        print(f"  - 阳性比例: {sum(1 for i in code15_samples if self.labels[i] > 0) / len(code15_samples) * 100:.2f}%")

        # 合并所有样本并随机打乱
        balanced_indices = samitrop_samples + ptbxl_samples + code15_samples
        random.shuffle(balanced_indices)

        print(f"平衡后总样本数: {len(balanced_indices)}")
        print(f"  - 阳性样本: {sum(1 for i in balanced_indices if self.labels[i] > 0)}")
        print(f"  - 阴性样本: {sum(1 for i in balanced_indices if self.labels[i] == 0)}")

        return balanced_indices

    def get_pos_weight(self):
        return torch.tensor(self.pos_weight, dtype=torch.float32)  # 返回正样本权重的PyTorch张量，用于BCEWithLogitsLoss等损失函数

    def get_sample_weights(self):
        """返回所有样本的权重，用于WeightedRandomSampler"""
        # 为balanced_indices中的样本返回对应的权重
        return [self.sample_weights[i] for i in self.balanced_indices]

    def __len__(self):
        return len(self.balanced_indices)  # 返回数据集的大小，即平衡后的记录数量，这是PyTorch Dataset类需要实现的方法。

    def __getitem__(self, idx):
        self._current_idx = idx  # 更新当前索引
        real_idx = self.balanced_indices[idx]  # 获取指定索引的样本,将传入的索引映射到平衡后的真实索引
        record = self.records[real_idx]  # 获取对应的记录ID和完整路径

        # 获取原始记录ID和文件夹
        folder = self.folder_map[record]['folder']
        original_id = self.folder_map[record]['original_id']
        record_path = os.path.join(folder, original_id)

        # 加载ECG信号
        try:
            signal, fields = load_signals(record_path)
        except Exception:
            try:  # 如果加载失败，尝试使用wfdb直接加载
                record_data = wfdb.rdrecord(record_path)
                signal = record_data.p_signal
            except Exception as e:  # 如果仍然失败，返回全零信号作为替代，保证数据加载不会中断
                print(f"Error loading record {record}: {e}")
                signal = np.zeros((self.max_length, 12), dtype=np.float32)

        signal = self._preprocess_signal(signal)
        label = self.labels[real_idx]  # 获取标签, 获取人口统计学特征
        age = self.ages[real_idx]
        sex = self.sexes[real_idx]
        dlabel = self.dlabels[real_idx]  # 获取领域标签

        # 不再使用源信息，只使用年龄和性别作为元数据
        metadata = np.array([age, sex], dtype=np.float32)

        if self.transform:
            signal = self.transform(signal)
        # 转换为PyTorch张量
        if isinstance(signal, np.ndarray):
            signal = torch.tensor(signal, dtype=torch.float32)

        metadata = torch.tensor(metadata, dtype=torch.float32)
        label = torch.tensor(1.0 if label else 0.0, dtype=torch.float32)
        dlabel = torch.tensor(dlabel, dtype=torch.float32)  # 将领域标签转换为张量

        # 返回信号、元数据、标签和领域标签
        return signal, metadata, label, dlabel

    def _preprocess_signal(self, signal):
        # 获取当前记录的来源
        source = 'unknown'
        if hasattr(self, '_current_idx') and self._current_idx < len(self.balanced_indices):
            real_idx = self.balanced_indices[self._current_idx]
            if real_idx < len(self.source_labels):
                source = self.source_labels[real_idx]

        if len(signal.shape) == 1:
            signal = np.expand_dims(signal, axis=1)  # 如果是一维数据，扩展维度以匹配12导联格式
        signal_12leads = signal

        # 如果是PTB-XL数据，需要重采样
        if source == 'PTB-XL':
            # 从500Hz降采样到400Hz
            original_len = signal_12leads.shape[0]
            target_len = int(original_len * 400 / 500)

            # 为每个导联创建一个重采样的结果
            resampled_signal = np.zeros((target_len, 12), dtype=signal_12leads.dtype)
            # 对每个导联分别进行重采样
            for lead in range(12):
                indices = np.linspace(0, original_len - 1, target_len)
                resampled_signal[:, lead] = np.interp(indices, np.arange(original_len), signal_12leads[:, lead])

            signal_12leads = resampled_signal

        # 处理信号长度
        if signal_12leads.shape[0] > self.max_length:
            # 如果信号过长，取中间部分
            start = (signal_12leads.shape[0] - self.max_length) // 2
            signal_12leads = signal_12leads[start:start + self.max_length, :]
        elif signal_12leads.shape[0] < self.max_length:
            # 如果信号过短，填充零
            padding = np.zeros((self.max_length - signal_12leads.shape[0], 12), dtype=signal_12leads.dtype)
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
        return signal_12leads.transpose(1, 0)  # 输出形状为 [12, seq_len]