# config.py

device = "cuda:3"  # 默认使用cuda:3
# 模型参数
num_classes = 2          # 类别数

# 训练参数
batch_size = 32

epochs=40

lr = 5e-5             # 基础学习率
weight_decay=5e-3

factor = 0.2  # 新的学习率是旧的学习率乘以该因子。
patience = 5  # 在监测量多少个周期没有改善后调整学习率。

# 其他配置
seed = 42
num_workers = 4
dropout = 0.2