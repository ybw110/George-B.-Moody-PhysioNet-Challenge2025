import torch
import torch.nn as nn
import torch.nn.functional as F
import config
class SEAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=16, stride=1, padding=7, se_reduction=16):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # 如果输入和输出通道数不一致，调整通道数
        self.skip_connection = nn.Sequential()
        if in_channels != out_channels:
            self.skip_connection = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1)

        # 添加 SE Attention 模块
        self.se_attention = SEAttention(out_channels, reduction=se_reduction)

    def forward(self, x):
        # 跳跃连接
        residual = self.skip_connection(x)

        # 主分支
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        # 调整 residual 的大小以匹配 out
        if residual.size(2) != out.size(2):
            residual = F.interpolate(residual, size=out.size(2), mode='nearest')

        # SE Attention
        out = self.se_attention(out)

        # 合并主分支和跳跃连接
        out += residual
        out = self.relu(out)
        return out


class SECNN(nn.Module):
    def __init__(self, input_length=4096, num_classes=1, include_metadata=True, metadata_dim=2):
        super(SECNN, self).__init__()

        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        self.include_metadata = include_metadata

        # 初始卷积层
        self.conv1 = nn.Conv1d(1, 64, kernel_size=16, stride=2, padding=7)  # 使用步长为2降低序列长度
        # self.conv1 = nn.Conv1d(12, 64, kernel_size=16, stride=2, padding=7)       # 使用12导联

        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)  # 进一步降低序列长度

        # 定义残差块，增加深度和降采样以处理更长的序列
        self.residual_blocks = nn.Sequential(
            ResidualBlock(64, 64, se_reduction=16),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 降采样
            nn.Dropout(p=config.dropout),

            ResidualBlock(64, 128, se_reduction=16),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 降采样
            nn.Dropout(p=config.dropout),

            ResidualBlock(128, 256, se_reduction=16),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 降采样
            nn.Dropout(p=config.dropout),

            ResidualBlock(256, 512, se_reduction=16),
            nn.AdaptiveAvgPool1d(1)  # 全局平均池化到固定长度
        )

        # 直接设置展平后的特征数量为512（因为AdaptiveAvgPool1d(1)会将每个通道池化为1个值）
        self.in_features = 512


        # 元数据处理（年龄和性别）
        if include_metadata:
            self.metadata_fc = nn.Sequential(
                nn.Linear(metadata_dim, 16),
                nn.ReLU(),
                nn.Dropout(p=config.dropout)
            )
            total_features = self.in_features + 16
        else:
            total_features = self.in_features

        # 分类头 - 修改为输出单个值
        self.classifier = nn.Sequential(
            nn.Linear(total_features, 64),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )


    def forward(self, x, metadata=None):
        # x shape: [batch_size, 1, sequence_length]
        # 确保输入在正确的设备上
        x = x.to(self.device)
        # 确保元数据也在正确的设备上
        if self.include_metadata and metadata is not None:
            metadata = metadata.to(self.device)

        x = self.conv1(x)  # 12导联的话，这里的 x 形状应该是 (batch_size, 12, 4096)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.residual_blocks(x)
        features = x.view(x.size(0), -1)  # 展平，shape: [batch_size, 512]

        # 如果包含元数据，处理元数据并连接
        if self.include_metadata and metadata is not None:
            metadata_features = self.metadata_fc(metadata)
            combined_features = torch.cat((features, metadata_features), dim=1)
        else:
            combined_features = features

        # 分类
        logits = self.classifier(combined_features)
        logits = logits.squeeze(1)

        return logits, features