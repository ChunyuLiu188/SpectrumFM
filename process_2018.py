import h5py
import numpy as np
from tqdm import tqdm
def min_max_normalize(data, min_value=None, max_value=None):
    min_val = np.min(data) if min_value is None else min_value
    max_val = np.max(data) if max_value is None else max_value
    return (data - min_val) / (max_val - min_val)
# 输入和输出文件路径
input_file = 'Data/GOLD_XYZ_OSC.0001_1024.hdf5'
output_file = 'Data/processed_2018_.h5'

# 每批次处理 4096 条数据
batch_size = 4096

# 打开输入文件以读取模式
with h5py.File(input_file, 'r') as f_in:
    # 获取原始数据集的形状信息
    total_samples = f_in['X'].shape[0]  # 假设 X、Y、Z 的形状相同
    
    # 计算处理后的总样本数
    total_new_samples = total_samples * 8
    
    # 创建输出文件
    with h5py.File(output_file, 'w') as f_out:
        # 为 X、Y 和 Z 创建新的输出数据集
        dset_X = f_out.create_dataset('X', shape=(total_new_samples, 128, 2), dtype=np.float32)
        dset_Y = f_out.create_dataset('Y', shape=(total_new_samples, 1), dtype=np.int64)
        dset_Z = f_out.create_dataset('Z', shape=(total_new_samples, 1), dtype=np.int64)

        # 批量处理数据
        for start_idx in tqdm(range(0, total_samples, batch_size)):
            end_idx = min(start_idx + batch_size, total_samples)
            
            # 读取当前批次的 X、Y、Z 数据
            X_batch = f_in['X'][start_idx:end_idx]  # Shape: (batch_size, 1024, channels)
            Y_batch = f_in['Y'][start_idx:end_idx]
            Z_batch = f_in['Z'][start_idx:end_idx]
            
            
            sample = X_batch.reshape(-1, 128, 2)
        
            I = sample[:, :, 0]
            Q = sample[:, :, 1]
        
            # 计算幅度
            amplitude = np.sqrt(I**2 + Q**2)
            amplitude = np.log10(amplitude)
        
            # 计算相位
            phase = np.arctan2(Q, I)
       
        
       
            sample = np.stack((amplitude, phase), axis=-1)
    

# 对每个 128x2 子矩阵中的 I 和 Q 分别进行归一化
            normalized_sample = np.zeros_like(sample)

            for i in range(sample.shape[0]):
                # 提取 I 和 Q
                I = sample[i, :, 0]
                Q = sample[i, :, 1]
                
                # 归一化 I 和 Q
                normalized_I = min_max_normalize(I)
                normalized_Q = min_max_normalize(Q)
                
                # 将归一化后的 I 和 Q 放回原位置
                normalized_sample[i, :, 0] = normalized_I
                normalized_sample[i, :, 1] = normalized_Q
                
        
                
            label = np.repeat(Y_batch, 8, axis=0).argmax(axis=-1).reshape(-1, 1)
            snr = np.repeat(Z_batch, 8, axis=0).reshape(-1, 1)
               
        

          
            
           
            new_start_idx = start_idx * 8
            new_end_idx = end_idx * 8
                
                # 将处理后的数据写入新文件
            dset_X[new_start_idx:new_end_idx] = normalized_sample
            dset_Y[new_start_idx:new_end_idx] = label
            dset_Z[new_start_idx:new_end_idx] = snr

print("数据处理完成并成功保存到新文件。")
