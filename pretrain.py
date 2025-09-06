import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from torch.utils.data import Dataset, DataLoader
from Model.model import ConformerPretrain
from dataset import HDF5Dataset2
from utils import create_lr_lambda, EarlyStopping, create_mask, masked_reconstruction_loss, add_noise
import pickle
from Args.args import parse_args
from tqdm import tqdm
import h5py
from torchsummary import summary
import os


def setup(rank, world_size):
    """初始化分布式训练环境"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # 初始化进程组
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    """清理分布式训练环境"""
    dist.destroy_process_group()


def train_single_epoch(model, train_dataloader, optimizer, scheduler, device, rank=0):
    """训练一个epoch"""
    model.train()
    train_loss = []
    loss_plot = []
    
    # 只在主进程显示进度条
    iterator = tqdm(enumerate(train_dataloader), disable=(rank != 0), 
                   total=len(train_dataloader), desc="Training")
    
    for idx, (data, masked_data, mask, pre_labels, label, snr) in iterator:
        data = data.to(device, non_blocking=True)
        masked_data = masked_data.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        pre_labels = pre_labels.to(device, non_blocking=True)
        
        reconstructed_signal, predicted_signal = model(masked_data, mask)
        reconstructed_loss = masked_reconstruction_loss(data, reconstructed_signal, mask, reduction='mean')
        predicted_loss = F.mse_loss(predicted_signal, pre_labels)
        loss = reconstructed_loss + predicted_loss
        
        train_loss.append(loss.item())
        loss_plot.append(loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # 更新进度条
        if rank == 0:
            iterator.set_postfix({'loss': f'{loss.item():.6f}'})
    
    return train_loss, loss_plot


def validate(model, val_dataloader, device, rank=0):
    """验证模型"""
    model.eval()
    val_loss = []
    
    iterator = tqdm(enumerate(val_dataloader), disable=(rank != 0), 
                   total=len(val_dataloader), desc="Validation")
    
    with torch.no_grad():
        for idx, (data, masked_data, mask, pre_labels, label, snr) in iterator:
            data = data.to(device, non_blocking=True)
            masked_data = masked_data.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            pre_labels = pre_labels.to(device, non_blocking=True)
            
            reconstructed_signal, predicted_signal = model(masked_data, mask)
            reconstructed_loss = masked_reconstruction_loss(data, reconstructed_signal, mask, reduction='mean')
            predicted_loss = F.mse_loss(predicted_signal, pre_labels)
            loss = reconstructed_loss + predicted_loss
            val_loss.append(loss.item())
    
    return val_loss


def train_distributed(rank, world_size, args):
    """分布式训练主函数"""
    setup(rank, world_size)
    
    # 设置设备
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    
    if rank == 0:
        print(f"Using distributed training with {world_size} GPUs")
    
    # 创建数据集
    train_dataset = HDF5Dataset2("Data/processed_2018.h5")
    val_dataset = HDF5Dataset2("Data/pre_val.h5")
    
    # 创建分布式采样器
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    
    # 创建数据加载器
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=256,  # 根据GPU数量调整批次大小
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=256,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # 创建模型
    model = ConformerPretrain(
        args.input_dim, 
        args.hidden_dim, 
        args.nhead, 
        args.num_layers, 
        args.dim_feedforward
    ).to(device)
    
    # 包装为DDP模型
    model = DDP(model, device_ids=[rank], output_device=rank)
    
    # 优化器和调度器
    optimizer = optim.AdamW(model.parameters(), lr=args.peak_lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, create_lr_lambda(args.warmup_steps, args.total_steps))
    
    # 早停机制（只在主进程使用）
    early_stopping = EarlyStopping(patience=5, verbose=True, path=args.pretrain_save_path) if rank == 0 else None
    
    all_loss_plot = []
    
    for epoch in range(args.epochs):
        # 设置采样器的epoch，确保每个epoch的shuffle是不同的
        train_sampler.set_epoch(epoch)
        
        # 训练
        train_loss, loss_plot = train_single_epoch(model, train_dataloader, optimizer, scheduler, device, rank)
        all_loss_plot.extend(loss_plot)
        
        # 收集所有进程的训练损失
        train_loss_tensor = torch.tensor(np.mean(train_loss), device=device)
        dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
        avg_train_loss = train_loss_tensor.item() / world_size
        
        if rank == 0:
            print(f"Epoch [{epoch+1}/{args.epochs}], TrainLoss: {avg_train_loss:.7f}")
        
        # 验证（可选，取消注释以启用）
        # val_loss = validate(model, val_dataloader, device, rank)
        # val_loss_tensor = torch.tensor(np.mean(val_loss), device=device)
        # dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
        # avg_val_loss = val_loss_tensor.item() / world_size
        
        # if rank == 0:
        #     print(f"Epoch [{epoch+1}/{args.epochs}], ValLoss: {avg_val_loss:.7f}")
        #     
        #     # 早停检查
        #     early_stopping(avg_val_loss, model.module)
        #     if early_stopping.early_stop:
        #         print("Early stopping")
        #         break
    
    # 只在主进程保存模型和损失数据
    if rank == 0:
        np.save('loss_plot.npy', np.array(all_loss_plot))
        torch.save(model.module.encoder.state_dict(), 'Checkpoint/pretrain_model_2018.pt')
        print("Training completed. Model and loss data saved.")
    
    cleanup()


def train_single_gpu(args):
    """单GPU训练函数"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using single device: {device}")
    
    # 创建数据集和数据加载器
    train_dataset = HDF5Dataset2("Data/processed_2018.h5")
    val_dataset = HDF5Dataset2("Data/pre_val.h5")
    
    train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=args.num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=args.num_workers)
    
    # 创建模型
    model = ConformerPretrain(
        args.input_dim, 
        args.hidden_dim, 
        args.nhead, 
        args.num_layers, 
        args.dim_feedforward
    ).to(device)
    
    # 优化器和调度器
    optimizer = optim.AdamW(model.parameters(), lr=args.peak_lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, create_lr_lambda(args.warmup_steps, args.total_steps))
    
    # 早停机制
    early_stopping = EarlyStopping(patience=5, verbose=True, path=args.pretrain_save_path)
    
    loss_plot = []
    
    for epoch in range(args.epochs):
        # 训练
        train_loss, epoch_loss_plot = train_single_epoch(model, train_dataloader, optimizer, scheduler, device)
        loss_plot.extend(epoch_loss_plot)
        
        print(f"Epoch [{epoch+1}/{args.epochs}], TrainLoss: {np.mean(train_loss):.7f}")
        
        # 验证（可选，取消注释以启用）
        # val_loss = validate(model, val_dataloader, device)
        # print(f"Epoch [{epoch+1}/{args.epochs}], ValLoss: {np.mean(val_loss):.7f}")
        # 
        # early_stopping(np.mean(val_loss), model)
        # if early_stopping.early_stop:
        #     print("Early stopping")
        #     break
    
    # 保存模型和损失数据
    np.save('loss_plot.npy', np.array(loss_plot))
    torch.save(model.encoder.state_dict(), 'Checkpoint/pretrain_model_2018.pt')
    print("Training completed. Model and loss data saved.")


def main():
    args = parse_args()
    
    # 检查是否有多个GPU可用
    if torch.cuda.device_count() > 1:
        print(f"Found {torch.cuda.device_count()} GPUs. Using distributed training.")
        world_size = torch.cuda.device_count()
        
        # 启动多进程分布式训练
        mp.spawn(train_distributed, args=(world_size, args), nprocs=world_size, join=True)
    else:
        print("Using single GPU/CPU training.")
        train_single_gpu(args)


if __name__ == "__main__":
    main()