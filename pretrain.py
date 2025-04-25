import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from Model.model import  ConformerPretrain
from dataset import HDF5Dataset2
from utils import create_lr_lambda, EarlyStopping, create_mask, masked_reconstruction_loss, add_noise
import pickle
from Args.args import parse_args
from tqdm import tqdm
import h5py
from torchsummary import summary
args = parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

train_dataset = HDF5Dataset2("Data/pre_train.h5")
val_dataset = HDF5Dataset2("Data/pre_val.h5")
model = ConformerPretrain(args.input_dim, args.hidden_dim, args.nhead, args.num_layers, args.dim_feedforward).to(device)
summary(model, input_size=(256, 128, 2))
optimizer = optim.AdamW(model.parameters(), lr=args.peak_lr)
train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=args.num_workers)
val_dataloader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=args.num_workers)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, create_lr_lambda(args.warmup_steps, args.total_steps))
early_stopping = EarlyStopping(patience=5, verbose=True, path=args.pretrain_save_path)
for epoch in range(15):
    model.train()
    train_loss = []
    for idx, (data, masked_data, mask, pre_labels, label, snr) in tqdm(enumerate(train_dataloader)):
        data, masked_data, mask, pre_labels = data.to(device), masked_data.to(device), mask.to(device), pre_labels.to(device) 
        reconstructed_signal, predicted_signal = model(masked_data, mask)
        reconstructed_loss = masked_reconstruction_loss(data, reconstructed_signal, mask, reduction='mean')
        predicted_loss = F.mse_loss(predicted_signal, pre_labels)
        loss = reconstructed_loss + predicted_loss
        train_loss.append(loss.item())
        # print(f"Loss: {loss.item():.7f}")
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        
        # 输出当前epoch的损失
    print(f"Epoch [{epoch+1}/{args.epochs}], TrainLoss: {np.mean(train_loss):.7f}")
    
    model.eval()
    val_loss = []
    with torch.no_grad():
        for idx, (data, masked_data, mask, pre_labels, label, snr) in enumerate(val_dataloader):
            data, masked_data, mask, pre_labels = data.to(device), masked_data.to(device), mask.to(device), pre_labels.to(device) 
            reconstructed_signal, predicted_signal = model(masked_data, mask)
            reconstructed_loss = masked_reconstruction_loss(data, reconstructed_signal, mask, reduction='mean')
            predicted_loss = F.mse_loss(predicted_signal, pre_labels)
            loss = reconstructed_loss + predicted_loss
            val_loss.append(loss.item())
    print(f"Epoch [{epoch+1}/{args.epochs}], ValLoss: {np.mean(val_loss):.7f}")
    
    early_stopping(np.mean(val_loss), model)

    # 检查是否应该提前停止
    if early_stopping.early_stop:
        print("Early stopping")
        break
    
