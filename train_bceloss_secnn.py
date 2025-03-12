# train_bceloss_secnn.py - 简化版SECNN训练脚本

import numpy as np
import random
import time
import torch
import json
import os
import pandas as pd
from datetime import datetime
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from model_SECNN_BCEloss import SECNN
from losses import FocalLoss
import config
from dataloader_with_3domains import ChagasECGDataLoader
from helper_code import compute_challenge_score, compute_auc, compute_accuracy, compute_f_measure



def set_seed(seed_value=42):
    """设置所有随机种子以确保实验的可重复性。"""
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_metrics(metrics, save_dir, filename):
    """保存训练指标到文件"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    filepath = os.path.join(save_dir, filename)
    df = pd.DataFrame(metrics)
    df.to_csv(filepath, index=False)
    print(f"Metrics saved to {filepath}")


def evaluate_predictions(y_true, y_pred_prob, y_pred_binary=None):
    if y_pred_binary is None:
        y_pred_binary = (y_pred_prob > 0.5).astype(int)
    # 计算指标
    challenge_score = compute_challenge_score(y_true, y_pred_prob)
    auroc, auprc = compute_auc(y_true, y_pred_prob)
    accuracy = compute_accuracy(y_true, y_pred_binary)
    f_measure = compute_f_measure(y_true, y_pred_binary)

    return {
        'challenge_score': challenge_score,
        'auroc': auroc,
        'auprc': auprc,
        'accuracy': accuracy,
        'f_measure': f_measure
    }


def train_secnn_model(data_folder, model_folder, epochs=20, verbose=True):
    # 设置随机种子
    set_seed(config.seed)

    if verbose:
        print("Starting SECNN model training...")

    # 创建时间戳
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # 创建带时间戳的模型保存目录
    model_save_dir = os.path.join(model_folder, f'chagas_model_{timestamp}')
    os.makedirs(model_save_dir, exist_ok=True)
    # 创建日志文件夹
    log_dir = os.path.join(model_save_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)

    # 创建指标记录字典
    metrics = {
        'epoch': [],
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'challenge_score': [],
        'auroc': [],
        'auprc': [],
        'accuracy': [],
        'f_measure': [],
        'time_elapsed': []
    }

    # 设置设备
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"Using {device} device")

    # 找到所有数据文件夹
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
    # 准备数据加载器
    loader = ChagasECGDataLoader(
        data_folders=data_folders,
        batch_size=config.batch_size,
        max_length=4096,
        num_workers=config.num_workers,
        pin_memory=True,
        seed=config.seed,
        source_weights=source_weights,
        resample_rate=resample_rate
    )

    # 获取训练/验证拆分
    train_loader, val_loader, train_dataset, val_dataset = loader.get_train_val_split(
        val_size=0.3,
        balance=True  # 平衡类别
    )

    if verbose:
        print(f"Data loaded: {len(train_dataset)} training samples, {len(val_dataset)} validation samples")

    # 初始化模型
    model = SECNN(
        input_length=4096,
        num_classes=config.num_classes,
        include_metadata=True
    ).to(device)

    # 损失函数和优化器     # 学习率调度器
    # criterion = nn.CrossEntropyLoss()

    # 获取正样本权重,对于二分类问题，特别是带有样本权重的情况，使用BCEWithLogitsLoss可能更适合
    pos_weight = train_dataset.get_pos_weight().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.Adam(model.parameters(), lr=config.lr,weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=config.factor,patience=config.patience, verbose=True)

    # 用于早停的变量
    best_challenge_score = 0
    patience_counter = 0
    patience = 20

    # 训练循环
    for epoch in range(epochs):
        if verbose:
            print(f"\n====== EPOCH {epoch + 1}/{epochs} ======")

        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0    # 记录正确预测的数量
        train_total = 0      # 记录总样本数
        start_time = time.time()

        for data in tqdm(train_loader, desc=f"Training (Epoch {epoch + 1}/{epochs})",
                         disable=not verbose):
            signals, metadata, labels, _ = data
            signals = signals.to(device).float()  # 添加 .float() 转换
            metadata = metadata.to(device).float() if metadata is not None else None
            # labels = labels.to(device).long()   # 对于CrossEntropyLoss，标签应为long类型
            labels = labels.to(device).float()  # 对于BCEWithLogitsLoss，标签应为float类型

            # 前向传播
            optimizer.zero_grad()
            logits, _ = model(signals, metadata)

            # 计算训练准确率
            # _, predicted = torch.max(logits.data, 1)
            # train_total += labels.size(0)
            # train_correct += (predicted == labels).sum().item()

            #在使用 CrossEntropyLoss 时，你的模型输出多个类别的概率（logits），但使用 BCEWithLogitsLoss 时，你需要一个单一的输出
            predicted = (torch.sigmoid(logits) > 0.5).float()
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            # 计算损失
            loss = criterion(logits, labels)

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # 平均训练损失和准确率
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []
        val_correct = 0    # 记录验证集正确预测的数量
        val_total = 0      # 记录验证集总样本数

        with torch.no_grad():
            for data in tqdm(val_loader, desc="Validating", disable=not verbose):
                signals, metadata, labels, _ = data
                signals = signals.to(device).float()  # 添加 .float() 转换
                metadata = metadata.to(device).float() if metadata is not None else None
                # labels = labels.to(device).long()
                labels = labels.to(device).float()  # 对于BCEWithLogitsLoss，标签应为float类型
                # 前向传播
                logits, _ = model(signals, metadata)

                # 计算验证准确率
                # _, predicted = torch.max(logits.data, 1)
                # val_total += labels.size(0)
                # val_correct += (predicted == labels).sum().item()

                # 修改为
                predicted = (torch.sigmoid(logits) > 0.5).float()
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                # 计算损失
                loss = criterion(logits, labels)
                val_loss += loss.item()

                # 计算预测概率
                # probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()   #这是针对多类别分类的。
                probs = torch.sigmoid(logits).cpu().numpy()    #如果使用BCEWithLogitsLoss，则应该使用sigmoid：

                val_preds.extend(probs)
                val_labels.extend(labels.cpu().numpy())

        # 平均验证损失和准确率
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total

        # 计算验证指标
        val_labels = np.array(val_labels)
        val_preds = np.array(val_preds)
        val_binary_preds = (val_preds > 0.5).astype(int)

        # 使用官方评估指标
        eval_metrics = evaluate_predictions(val_labels, val_preds, val_binary_preds)
        challenge_score = eval_metrics['challenge_score']
        auroc = eval_metrics['auroc']
        auprc = eval_metrics['auprc']
        accuracy = eval_metrics['accuracy']
        f_measure = eval_metrics['f_measure']

        # 更新学习率
        scheduler.step(challenge_score)

        # 记录时间
        time_elapsed = time.time() - start_time

        # 记录指标
        metrics['epoch'].append(epoch + 1)
        metrics['train_loss'].append(train_loss)
        metrics['train_acc'].append(train_acc)  # 添加训练准确率
        metrics['val_loss'].append(val_loss)
        metrics['val_acc'].append(val_acc)     # 使用官方计算的准确率

        metrics['challenge_score'].append(challenge_score)
        metrics['auroc'].append(auroc)
        metrics['auprc'].append(auprc)
        metrics['accuracy'].append(accuracy)
        metrics['f_measure'].append(f_measure)
        metrics['time_elapsed'].append(time_elapsed)

        if verbose:
            print(f"Epoch {epoch + 1}/{epochs}, " +
                  f"Train Loss: {train_loss:.4f}, " +
                  f"Train Acc: {train_acc:.4f}, " +  # 添加训练准确率输出
                  f"Val Loss: {val_loss:.4f}, " +
                  f"Val Acc: {val_acc:.4f}, " +
                  f"Challenge Score: {challenge_score:.4f}, " +
                  f"AUROC: {auroc:.4f}, AUPRC: {auprc:.4f}, " +
                  f"Accuracy: {accuracy:.4f}, F-measure: {f_measure:.4f}, " +
                  f"Time: {time_elapsed:.2f}s")

        # 早停检查
        if challenge_score > best_challenge_score:
            best_challenge_score = challenge_score
            # 保存最佳模型
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch + 1,
                'best_challenge_score': best_challenge_score,
                'train_acc': train_acc,  # 保存训练准确率
                'val_acc': val_acc,  # 保存验证准确率
                'auroc': auroc,
                'auprc': auprc,
                'accuracy': accuracy,
                'f_measure': f_measure,
                # 不要直接保存 config 对象，而是保存必要的配置参数
            'config_params': {
                'device': config.device,
                'num_classes': config.num_classes,
                'batch_size': config.batch_size,
                'lr': config.lr,
                'weight_decay': config.weight_decay,
                'dropout': config.dropout,
                'seed': config.seed,
                # 添加其他需要的配置参数
            }
        }
            torch.save(checkpoint, os.path.join(model_save_dir, 'best_model.pth'))
            patience_counter = 0

            if verbose:
                print(f"新的最佳模型已保存！Challenge分数: {challenge_score:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch + 1}")
                break

        # 每个epoch保存一次指标
        save_metrics(metrics, log_dir, f'metrics_epoch_{epoch + 1}_{timestamp}.csv')

    # 保存完整指标记录
    save_metrics(metrics, log_dir, f'complete_metrics_{timestamp}.csv')

    # 保存最终结果
    final_results = {
        'best_challenge_score': best_challenge_score,
        'timestamp': timestamp,
        'total_epochs': epoch + 1,
        'early_stop': patience_counter >= patience
    }

    with open(os.path.join(model_save_dir, 'final_results.json'), 'w') as f:
        json.dump(final_results, f, indent=4)

    if verbose:
        print(f"训练完成。最佳Challenge分数: {best_challenge_score:.4f}")
        print(f"模型已保存到 {model_save_dir}")

    return {
        'model': model,
        'save_dir': model_save_dir,
        'metrics': metrics
    }

