import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
from torch.utils.data import Dataset, DataLoader
import h5py
from Args.args import parse_args
from tqdm import tqdm
import h5py
from torchsummary import summary
from utils import standardize_IQ, iq2ap, add_noise, create_mask
args = parse_args()
class IQDataset(Dataset):
    def __init__(self, datas, labels, snrs):
        super(IQDataset, self).__init__()
        self.datas = torch.from_numpy(datas)
        self.labels = torch.from_numpy(labels)
        self.snrs = torch.from_numpy(snrs)
        
    def __len__(self):
        return len(self.datas)
    
    def __getitem__(self, idx):
        return self.datas[idx], self.labels[idx], self.snrs[idx]
    
    
class AMCDataset(Dataset):
    def __init__(self, data_dict):
        super(AMCDataset, self).__init__()
        self.values = torch.from_numpy(data_dict["value"])
        self.labels = torch.from_numpy(data_dict["label"])
        # 如果你还需要 snr，可以在此添加
        if "snr" in data_dict:
            self.snrs = torch.from_numpy(data_dict["snr"])
        else:
            self.snrs = None
        
    def __len__(self):
        return len(self.values)
    
    def __getitem__(self, idx):
        if self.snrs is not None:
            return self.values[idx], self.labels[idx], self.snrs[idx]
        else:
            return self.values[idx], self.labels[idx]


class HDF5Dataset(Dataset):
    def __init__(self, hdf5_file, dataset_name, labels=None):
        self.hdf5_file = hdf5_file
        self.dataset_name = dataset_name
        self.labels = labels
        self.file = h5py.File(self.hdf5_file, "r")  # 初始化时打开文件

    def __del__(self):
        # 确保文件在实例销毁时关闭
        self.file.close()
    def __len__(self):
        # 返回数据集的长度
        return len(self.file[self.dataset_name])

    def __getitem__(self, idx):
        sample = self.file[self.dataset_name][idx]
       

        
        
        # 返回无标签的数据
        return torch.tensor(sample, dtype=torch.float32)


class HDF5Dataset2(Dataset):
    def __init__(self, hdf5_file):
        self.hdf5_file = hdf5_file
        # 仅打开文件以获取数据长度
        with h5py.File(self.hdf5_file, "r") as f:
            self.data_length = len(f["X"])  # 获取样本数量

    def __len__(self):
        return self.data_length

    def __getitem__(self, idx):
        # 每次访问索引时才打开文件
        with h5py.File(self.hdf5_file, "r") as f:
            sample = f["X"][idx]  # 延迟加载样本
            label = f["Y"][idx]   # 延迟加载标签
            snr = f["Z"][idx]     # 延迟加载信噪比

        # 转换为 PyTorch 张量
        sample = torch.tensor(sample, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        snr = torch.tensor(snr, dtype=torch.long)
        noise_data = add_noise(sample, args.noise_std)
        masked_data, mask, pre_labels = create_mask(noise_data, args.mask_ratio)
        return sample, masked_data, mask, pre_labels, label, snr


class HDF5Dataset_train(Dataset):
    def __init__(self, hdf5_file):
        self.hdf5_file = hdf5_file
        # 仅打开文件以获取数据长度
        with h5py.File(self.hdf5_file, "r") as f:
            self.data_length = len(f["train_X"])  # 获取样本数量

    def __len__(self):
        return self.data_length

    def __getitem__(self, idx):
        # 每次访问索引时才打开文件
        with h5py.File(self.hdf5_file, "r") as f:
            sample = f["train_X"][idx]  # 延迟加载样本
            label = f["train_Y"][idx]   # 延迟加载标签
            snr = f["train_Z"][idx]     # 延迟加载信噪比

        # 转换为 PyTorch 张量
        sample = torch.tensor(sample, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
       
        return sample, label
       
class HDF5Dataset_test(Dataset):
    def __init__(self, hdf5_file):
        self.hdf5_file = hdf5_file
        # 仅打开文件以获取数据长度
        with h5py.File(self.hdf5_file, "r") as f:
            self.data_length = len(f["test_X"])  # 获取样本数量

    def __len__(self):
        return self.data_length

    def __getitem__(self, idx):
        # 每次访问索引时才打开文件
        with h5py.File(self.hdf5_file, "r") as f:
            sample = f["test_X"][idx]  # 延迟加载样本
            label = f["test_Y"][idx]   # 延迟加载标签
            snr = f["test_Z"][idx]     # 延迟加载信噪比

        # 转换为 PyTorch 张量
        sample = torch.tensor(sample, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
       
        return sample, label





