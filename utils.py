import torch
import torch.nn
import numpy as np
import torch
import numpy as np

import torch
import numpy as np

import torch

def create_mask(input_data, mask_ratio=0.3):
    """
    input_data: Tensor of shape [seq_length, input_dim]
    mask_ratio: The ratio of the sequence to mask (0 to 1)
    
    Returns:
        masked_input: The input with masked positions (shape [seq_length, input_dim])
        mask_matrix: Mask matrix with 1 for unmasked and 0 for masked positions (shape [seq_length])
        pre_label: The last element in the sequence (shape [input_dim])
    """
    seq, input_dim = input_data.shape  # For a single sample, shape is [seq_length, input_dim]
    
    # Create a mask matrix (1 for unmasked, 0 for masked)
    mask_matrix = torch.ones(seq).to(input_data.device)
    
    # Calculate the number of masked positions
    num_masked = int(seq * mask_ratio)
    
    # Randomly choose the indices to mask
    mask_indices = torch.randperm(seq)[:num_masked]
    mask_matrix[mask_indices] = 0
    
    # Ensure that the last element is not masked (for prediction)
    mask_matrix[-1] = 0
    
    # Apply the mask to the input
    masked_input = torch.multiply(input_data, mask_matrix.unsqueeze(-1))  # Broadcasting to match the shape
    
    # The pre_label is the last element in the sequence (used for prediction)
    pre_label = input_data[-1, :]
    
    return masked_input, mask_matrix, pre_label
 


def create_lr_lambda(warmup_steps, total_steps):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # Warm-up阶段：线性从0升到最大学习率
            return float(current_step) / float(max(1, warmup_steps))
        # Warm-up后，逐渐衰减，采用余弦退火策略
        return 0.5 * (1.0 + np.cos(np.pi * (current_step - warmup_steps) / (total_steps - warmup_steps)))
    
    return lr_lambda

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.0, verbose=True, path='checkpoint.pth', monitor='loss', pretrain=True):
        """
        :param patience: 在验证指标不改善的情况下允许的训练epoch数
        :param min_delta: 判断为指标改善的最小变化幅度
        :param verbose: 如果为True，则会输出每次改善时的信息
        :param path: 检查点文件路径，保存最优模型
        :param monitor: 监控的指标，'loss' 表示验证损失，'accuracy' 表示验证准确率
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.path = path
        self.monitor = monitor
        self.counter = 0
        self.best_value = None
        self.early_stop = False
        self.pretrain = pretrain

    def __call__(self, val_metric, model):
        """
        更新早停状态。
        :param val_metric: 当前epoch的验证指标值（可以是损失或准确率）
        :param model: 当前模型
        """
        if self.best_value is None:
            self.best_value = val_metric
            self.save_checkpoint(model)
        elif self._is_improvement(val_metric):
            self.best_value = val_metric
            self.counter = 0
            self.save_checkpoint(model)
            if self.verbose:
                print(f"Validation {self.monitor} improved. Saving model ...")
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    def _is_improvement(self, val_metric):
        """
        判断当前指标是否有改进。
        """
        if self.monitor == 'loss':
            return val_metric < self.best_value - self.min_delta
        elif self.monitor == 'accuracy':
            return val_metric > self.best_value + self.min_delta
        else:
            raise ValueError("Monitor must be 'loss' or 'accuracy'.")

    def save_checkpoint(self, model):
        if self.pretrain:
            torch.save(model.encoder.state_dict(), self.path)
        else:
            torch.save(model.state_dict(), self.path)


def standardize_IQ(data, I_min, I_max, Q_min, Q_max):
   
    
    I_channel = data[:, :, 0]
    Q_channel = data[:, :, 1]
    # 归一化 I 和 Q 通道到 [-1, 1] 范围
    I_channel_normalized = 2 * (I_channel - I_min) / (I_max - I_min) - 1
    Q_channel_normalized = 2 * (Q_channel - Q_min) / (Q_max - Q_min) - 1

    # 重新组合 I 和 Q 通道
    standardized_data = np.stack([I_channel_normalized, Q_channel_normalized], axis=-1)
    return standardized_data

def iq2ap(IQ_data):
    """
    Convert IQ 2 AP and conduct log scaling
    """
    # 提取I和Q分量
    I = IQ_data[:, :, 0]
    Q = IQ_data[:, :, 1]
    
    # 计算幅度
    amplitude = np.sqrt(I**2 + Q**2)
    amplitude = np.log(amplitude)
    
    # 计算相位
    phase = np.arctan2(Q, I)
    
    # 将幅度和相位组合成一个新的三维数组
    AP_data = np.stack((amplitude, phase), axis=-1)
    
    return AP_data
def min_max_normalize(data, min_value=None, max_value=None):
    min_val = np.min(data) if min_value is None else min_value
    max_val = np.max(data) if max_value is None else max_value
    return (data - min_val) / (max_val - min_val)
def normalize(sample):
    normalized_sample = np.zeros_like(sample)
    # max_a = -10000
    # max_p = -10000
    # min_a = 10000
    # min_p = 10000
    for i in range(sample.shape[0]):
        # 提取 I 和 Q
        I = sample[i, :, 0]
        Q = sample[i, :, 1]
        
        # 归一化 I 和 Q
        normalized_I = min_max_normalize(I)
        normalized_Q = min_max_normalize(Q)
        # normalized_I = min_max_normalize(I)
        # normalized_Q = min_max_normalize(Q)
        # 将归一化后的 I 和 Q 放回原位置
        normalized_sample[i, :, 0] = normalized_I
        normalized_sample[i, :, 1] = normalized_Q
    #     if np.max(Q) > max_p:
    #         max_p = np.max(Q)
    #     if np.min(Q) < min_p:
    #         min_p = np.min(Q)

    # # 更新最大和最小幅度
    #     if np.max(I) > max_a:
    #         max_a = np.max(I)
    #     if np.min(I) < min_a:
    #         min_a = np.min(I)
    # print('max_p:', max_p, 'min_p:', min_p, 'max_a:', max_a, 'min_a:', min_a)
    return normalized_sample

def masked_reconstruction_loss(y_true, y_pred, mask, reduction='mean'):
    """
    计算仅在掩码为0的区域的重构损失，直接提取这些区域。
    
    参数:
    - y_true: 真实标签 (tensor)，形状为 [batch_size, seq_length, 2]
    - y_pred: 预测输出 (tensor)，形状为 [batch_size, seq_length, 2]
    - mask: 掩码 (tensor)，形状为 [batch_size, seq_length]，1表示不计算损失，0表示计算损失
    - reduction: 损失的归约方式 ('mean' 或 'sum')
    
    返回:
    - loss: 重构损失 (scalar tensor)
    """
    # 将掩码扩展到与y_true和y_pred相同的维度
    mask_expanded = mask.unsqueeze(-1).expand_as(y_true)  # 形状变为 [batch_size, seq_length, 2]
    
    # 将掩码转换为布尔张量
    mask_bool = mask_expanded == 0
    
    # 使用布尔索引提取需要计算损失的元素
    y_true_masked = y_true[mask_bool]
    y_pred_masked = y_pred[mask_bool]
    
    if y_true_masked.numel() == 0:
        # 如果没有有效的元素用于计算损失，返回0或适当的值
        return torch.tensor(0., device=y_true.device)
    
    # 计算元素级平方差
    squared_difference = torch.pow(y_true_masked - y_pred_masked, 2)
    
    # 根据指定的reduction方法计算损失
    if reduction == 'mean':
        loss = squared_difference.mean()
    elif reduction == 'sum':
        loss = squared_difference.sum()
    else:
        raise ValueError("reduction must be one of 'mean' or 'sum'")
    
    return loss

def add_noise(x, std):
    noise = torch.randn_like(x, device=x.device) * std
    return x + noise
