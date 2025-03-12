#!/usr/bin/env python
import glob
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler, random_split
from sklearn.model_selection import KFold, train_test_split
import random
from dataset_with_3domains import MultiSourceChagasECGDataset
from data_aug import *


class ChagasECGDataLoader:

    def __init__(self, data_folders, batch_size=32, max_length=4096,
                 num_workers=4, pin_memory=True, seed=42,
                 source_weights=None, resample_rate=None):

        self.data_folders = data_folders if isinstance(data_folders, list) else [data_folders]
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.seed = seed
        self.source_weights = source_weights or {}  # 添加源权重
        self.resample_rate = resample_rate or {}  # 添加重采样率字典，用于处理不同采样率

        # 为了重现性设置随机种子
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        # 从所有文件夹中收集记录
        self.records = []
        self.folder_map = {}  # 记录ID到文件夹的映射

        for folder in self.data_folders:
            folder_name = os.path.basename(folder)

            # 确定数据源
            if 'code15' in folder_name.lower():
                source = 'CODE-15%'
            elif 'samitrop' in folder_name.lower():
                source = 'SaMi-Trop'
            elif 'ptb' in folder_name.lower():
                source = 'PTB-XL'
            else:
                source = 'unknown'

            # 收集记录路径
            record_paths = glob.glob(os.path.join(folder, "*.hea"))

            # 对于PTB-XL，需要递归搜索子文件夹
            if source == 'PTB-XL':
                for subdir in glob.glob(os.path.join(folder, "*")):
                    if os.path.isdir(subdir):
                        record_paths.extend(glob.glob(os.path.join(subdir, "*.hea")))

            folder_records = [os.path.basename(path).replace(".hea", "") for path in record_paths]

            # 为了避免ID冲突，为每个记录添加文件夹前缀
            prefixed_records = [f"{folder_name}_{record}" for record in folder_records]

            # 记录原始记录ID与文件夹的关系
            for original, prefixed in zip(folder_records, prefixed_records):
                record_dir = os.path.dirname(record_paths[folder_records.index(original)])
                self.folder_map[prefixed] = {
                    'folder': record_dir,
                    'original_id': original,
                    'source': source  # 添加数据源信息
                }
            self.records.extend(prefixed_records)

        print(f"Collected {len(self.records)} records from {len(self.data_folders)} folders")

        # 默认数据增强
        self.train_transform = Compose([
            Reshape(),  # 确保数据形状正确
            RandomAddGaussian(sigma=0.01),  # 添加高斯噪声
            RandomScale(sigma=0.1),  # 随机缩放
            RandomStretch(sigma=0.01),
            RandomCrop(crop_len=10),
            BaselineWander(max_wander=0.03, wander_step=0.005),  # 基线漂移
            Normalize(type="mean-std")  # 标准化
        ])

        self.val_transform = Compose([
            Reshape(),  # 确保数据形状正确
            Normalize(type="mean-std")  # 只对验证集和测试集进行标准化
        ])

    def set_transforms(self, train_transform=None, val_transform=None):
        if train_transform is not None:
            self.train_transform = train_transform
        if val_transform is not None:
            self.val_transform = val_transform

    def get_train_val_split(self, val_size=0.2, balance=True):
        # 为不同数据源设置权重
        source_weights = self.source_weights or {
            'CODE-15%': 0.8,  # CODE-15%的弱标签样本权重较低
            'SaMi-Trop': 1.5,  # SaMi-Trop样本权重较高
            'PTB-XL': 1.0,  # PTB-XL样本权重正常
        }

        # 创建支持多文件夹的数据集
        full_dataset = MultiSourceChagasECGDataset(
            records=self.records,
            folder_map=self.folder_map,
            max_length=self.max_length,
            transform=None,
            balance=balance,
            source_weights=source_weights
        )

        # 获取有效索引总数
        dataset_size = len(full_dataset)

        # 计算分割大小
        val_count = int(val_size * dataset_size)
        train_count = dataset_size - val_count

        # 随机分割
        train_dataset, val_dataset = random_split(
            full_dataset,
            [train_count, val_count],
            generator=torch.Generator().manual_seed(self.seed)
        )

        # 使用WeightedRandomSampler确保批次中的数据分布
        sample_weights = full_dataset.get_sample_weights()
        train_indices = train_dataset.indices
        train_weights = [sample_weights[i] for i in train_indices]

        # 创建加权采样器，replacement=True确保能够重复抽样少数类
        train_sampler = torch.utils.data.WeightedRandomSampler(
            weights=train_weights,
            num_samples=len(train_weights),
            replacement=True
        )

        # 为训练集和验证集创建数据加载器
        train_loader = DataLoader(
            full_dataset,
            batch_size=self.batch_size,
            sampler=train_sampler,  # 使用加权采样器
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            worker_init_fn=self._worker_init_fn,
            collate_fn=lambda x: self._custom_collate_fn(x, is_train=True)
        )

        val_loader = DataLoader(
            full_dataset,
            batch_size=self.batch_size,
            sampler=val_dataset.indices,  # 使用验证集索引
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            worker_init_fn=self._worker_init_fn,
            collate_fn=lambda x: self._custom_collate_fn(x, is_train=False)
        )

        print(f"Created data split:")
        print(f"  Training set: {len(train_dataset)} records")
        print(f"  Validation set: {len(val_dataset)} records")

        return train_loader, val_loader, full_dataset, val_dataset

    def _worker_init_fn(self, worker_id):
        """
        为每个工作线程设置不同的随机种子
        """
        worker_seed = self.seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    def _custom_collate_fn(self, batch, is_train=True):
        signals, metadata, labels, dlabels = [], [], [], []

        # 对批次中的每个样本应用适当的转换
        for signal, meta, label, dlabel in batch:
            # 应用合适的转换
            if is_train:
                signal = self.train_transform(signal)
            else:
                signal = self.val_transform(signal)

            # 确保信号是PyTorch张量
            if isinstance(signal, np.ndarray):
                signal = torch.tensor(signal, dtype=torch.float32)

            # 形状已经在_preprocess_signal中处理为[12, seq_len]，不需要额外处理

            signals.append(signal)
            metadata.append(meta)
            labels.append(label)
            dlabels.append(dlabel)

        # 将列表堆叠为批次张量
        signals = torch.stack(signals)
        metadata = torch.stack(metadata)
        labels = torch.stack(labels)
        dlabels = torch.stack(dlabels)

        return signals, metadata, labels, dlabels


# 运行函数部分
def run_sample_data():
    # 定义多个数据文件夹
    data_folders = [
        "/data1/ybw/挑战赛2025/processed_data/code15_part0_output",
        "/data1/ybw/挑战赛2025/processed_data/code15_part17_output",  # CODE-15%数据
        "/data1/ybw/挑战赛2025/processed_data/code15_part11_output",
        "/data1/ybw/挑战赛2025/processed_data/code15_part1_output",

        "/data1/ybw/挑战赛2025/processed_data/samitrop_output",  # SaMi-Trop数据
        "/data1/ybw/挑战赛2025/processed_data/PTB-XL-500"  # PTB-XL数据
    ]

    # 设置不同数据源的权重
    source_weights = {
        'CODE-15%': 0.8,  # CODE-15%弱标签权重低
        'SaMi-Trop': 1.5,  # SaMi-Trop强标签权重高
        'PTB-XL': 1.0  # PTB-XL正常权重
    }

    # 设置采样率转换信息
    resample_rate = {
        'PTB-XL': 500  # PTB-XL是500Hz，需要转换到400Hz
    }

    # 创建数据加载器
    loader = ChagasECGDataLoader(
        data_folders=data_folders,
        batch_size=32,
        max_length=4096,
        num_workers=0,  # 调试时使用单线程
        pin_memory=False,
        seed=42,
        source_weights=source_weights,
        resample_rate=resample_rate
    )

    # 获取训练/验证拆分，确保平衡
    train_loader, val_loader, train_dataset, val_dataset = loader.get_train_val_split(
        val_size=0.2,
        balance=True  # 启用平衡采样
    )

    # 检查数据形状和类型
    print("\n检查训练集样本批次:")
    for signals, metadata, labels, dlabels in train_loader:
        print(f"Signals shape: {signals.shape}")
        print(f"Metadata shape: {metadata.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Domain labels shape: {dlabels.shape}")

        # 检查第一个批次的标签分布
        pos_count = torch.sum(labels).item()
        print(f"Positive samples in batch: {pos_count}/{len(labels)} ({pos_count / len(labels) * 100:.2f}%)")

        # 只检查第一个批次
        break

    print("\n检查验证集样本批次:")
    for signals, metadata, labels, dlabels in val_loader:
        print(f"Signals shape: {signals.shape}")

        # 只检查第一个批次
        break

    print("\n数据加载测试成功完成!")


if __name__ == "__main__":
    # 执行示例运行函数
    run_sample_data()