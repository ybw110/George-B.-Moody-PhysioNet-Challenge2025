
import numpy as np
import random
from scipy.signal import resample


class Compose:
    """组合多个转换"""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, signal):
        for transform in self.transforms:
            signal_shape_before = signal.shape
            signal = transform(signal)
            signal_shape_after = signal.shape
            if signal_shape_before != signal_shape_after:
                print(
                    f"Transform {transform.__class__.__name__} changed shape from {signal_shape_before} to {signal_shape_after}")
        return signal


class Reshape(object):
    def __call__(self, seq):
        if len(seq.shape) == 1:                   # 如果是一维数组
            return np.expand_dims(seq, axis=0)    # 将一维数组转换为二维数组，形式为[1, 序列长度]
        return seq


class Retype(object):
    def __call__(self, seq):
        return seq.astype(np.float32)



class AddGaussian(object):
    def __init__(self, sigma=0.01):
        self.sigma = sigma

    def __call__(self, seq):
        return seq + np.random.normal(loc=0, scale=self.sigma, size=seq.shape)


class RandomAddGaussian(object):
    def __init__(self, sigma=0.01):
        self.sigma = sigma

    def __call__(self, seq):
        if np.random.randint(2):
            return seq
        else:
            return seq + np.random.normal(loc=0, scale=self.sigma, size=seq.shape)
'''作用：以一定的概率向序列数据中添加高斯噪声。这提供了一种随机性，可以进一步提高模型的泛化能力。
适用数据：同上，适用于需要提高对随机噪声鲁棒性的序列数据。'''

class Scale(object):
    def __init__(self, sigma=0.01):
        self.sigma = sigma

    def __call__(self, seq):
        scale_factor = np.random.normal(loc=1, scale=self.sigma, size=(seq.shape[0], 1))
        scale_matrix = np.matmul(scale_factor, np.ones((1, seq.shape[1])))
        return seq*scale_matrix
'''作用：随机缩放序列数据的各个维度。这种增强技术可以帮助模型学习到数据的尺度不变性。
适用数据：多维序列数据，特别是当数据的尺度变化对模型性能有影响时。'''

class RandomScale(object):
    def __init__(self, sigma=0.01):
        self.sigma = sigma

    def __call__(self, seq):
        if np.random.randint(2):
            return seq
        else:
            scale_factor = np.random.normal(loc=1, scale=self.sigma, size=(seq.shape[0], 1))
            scale_matrix = np.matmul(scale_factor, np.ones((1, seq.shape[1])))
            return seq*scale_matrix
'''作用：以一定的概率随机缩放序列数据的各个维度。
适用数据：同上，特别适合于需要模型学习尺度不变性的多维序列数据。'''

class RandomStretch(object):
    def __init__(self, sigma=0.3):
        self.sigma = sigma

    def __call__(self, seq):
        if np.random.randint(2):
            return seq
        else:
            seq_aug = np.zeros(seq.shape)
            len = seq.shape[1]
            length = int(len * (1 + (random.random()-0.5)*self.sigma))
            for i in range(seq.shape[0]):
                y = resample(seq[i, :], length)
                if length < len:
                    if random.random() < 0.5:
                        seq_aug[i, :length] = y
                    else:
                        seq_aug[i, len-length:] = y
                else:
                    if random.random() < 0.5:
                        seq_aug[i, :] = y[:len]
                    else:
                        seq_aug[i, :] = y[length-len:]
            return seq_aug
'''作用：随机拉伸或压缩序列数据。这种增强技术通过改变序列的长度，可以帮助模型更好地处理时间或空间维度上的变化。
适用数据：时间序列数据或任何一维序列数据，特别是长度变化对模型预测有重要影响时。'''


class RandomCrop(object):
    def __init__(self, crop_len=20):
        self.crop_len = crop_len

    def __call__(self, seq):
        if np.random.randint(2):
            return seq
        else:
            # 修复：检查序列长度是否大于crop_len
            if seq.shape[1] <= self.crop_len:
                # 如果序列长度小于或等于crop_len，则跳过裁剪
                return seq

            # 计算可选的随机索引范围（确保max_index > 0）
            max_index = seq.shape[1] - self.crop_len
            if max_index <= 0:
                return seq  # 安全检查，虽然前面已经检查过

            random_index = np.random.randint(max_index)
            seq[:, random_index:random_index + self.crop_len] = 0
            return seq
'''作用：在序列数据中随机选择一个区域并将其值设置为0，模拟数据的部分丢失情况。这可以增强模型处理不完整数据的能力。
适用数据：适用于任何序列数据，尤其是在实际应用中可能遇到数据缺失的场景。'''

class Normalize(object):
    def __init__(self, type="0-1"):  # "0-1", "1-1", "mean-std"
        self.type = type

    def __call__(self, seq):
        if self.type == "0-1":
            denom = seq.max() - seq.min()
            if denom == 0:
                seq = np.zeros_like(seq)
            else:
                seq = (seq - seq.min()) / denom
        elif self.type == "1-1":
            denom = seq.max() - seq.min()
            if denom == 0:
                seq = np.zeros_like(seq)
            else:
                seq = 2 * (seq - seq.min()) / denom - 1
        elif self.type == "mean-std":
            std = seq.std()
            if std == 0:
                seq = np.zeros_like(seq)
            else:
                seq = (seq - seq.mean()) / std
        else:
            raise ValueError('This normalization type is not supported!')
        return seq


'''作用：将序列数据标准化，有三种不同的方式："0-1"归一化到[0, 1]区间，"1-1"归一化到[-1, 1]区间，"mean-std"进行均值为0，标准差为1的标准化。
适用数据：几乎所有类型的序列数据'''




# 基线漂移
class BaselineWander:
    def __init__(self, max_wander=0.05, wander_step=0.01):
        self.max_wander = max_wander
        self.wander_step = wander_step

    def __call__(self, signal):
        # 获取信号的第一个维度大小（时间步长）
        time_steps = signal.shape[0]

        # 生成基线漂移的初始值
        wander = np.random.uniform(-self.max_wander, self.max_wander)

        # 为每个时间步长生成随机步长
        random_steps = np.random.uniform(-self.wander_step, self.wander_step, size=time_steps)

        # 计算累积漂移
        cum_wander = np.cumsum(random_steps)

        # 根据信号的维度重塑累积漂移
        if len(signal.shape) > 1:
            # 如果信号是多维的（例如多个通道），将累积漂移扩展以匹配信号形状
            cum_wander = cum_wander.reshape(-1, 1)

        return signal + wander + cum_wander

# 振幅抖动
class RandomAmplitudeScaling:
    def __init__(self, max_scaling=0.1):
        self.max_scaling = max_scaling

    def __call__(self, signal):
        scaling_factor = np.random.uniform(1 - self.max_scaling, 1 + self.max_scaling)
        return signal * scaling_factor

# 时间扭曲
class RandomTimeWarping:
    def __init__(self, max_warp=0.2):
        self.max_warp = max_warp

    def __call__(self, signal):
        warp_factor = np.random.uniform(1 - self.max_warp, 1 + self.max_warp)
        time_steps = np.arange(signal.size)
        new_time_steps = np.linspace(time_steps[0], time_steps[-1], num=int(signal.size * warp_factor))
        return np.interp(new_time_steps, time_steps, signal)

# 信号混合
class SignalMixing:
    def __init__(self, mix_ratio=0.1):
        self.mix_ratio = mix_ratio

    def __call__(self, signal):
        # 这里的另一个信号可以来自相同数据集的另一记录
        # signal_to_mix = ... (您需要实现获取另一个信号的逻辑)
        # 对于演示，我们只是简单地用零信号来混合
        signal_to_mix = np.zeros_like(signal)
        return (1 - self.mix_ratio) * signal + self.mix_ratio * signal_to_mix

# 随机重新采样（插值）
class RandomResampling:
    def __init__(self, new_length_factor=0.5):
        self.new_length_factor = new_length_factor

    def __call__(self, signal):
        new_length = int(len(signal) * np.random.uniform(1 - self.new_length_factor, 1 + self.new_length_factor))
        resampled_signal = np.interp(
            np.linspace(0, len(signal) - 1, num=new_length),
            np.arange(len(signal)),
            signal
        )
        return resampled_signal
