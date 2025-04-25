import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from Model.AMC_model import ModulationRecognitionGRU, AMC_Net, MSNet, ResNet, VGG, CNN2, DAE, CGDNN, MCNet, Transformer, GRU2
from Model.model import ConformerClassifier, LLMClassifier
# from Model.model import TransformerClassifierWithCLS
from dataset import AMCDataset
from utils import create_lr_lambda, EarlyStopping, standardize_IQ, iq2ap, normalize
import pickle
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm
from torchsummary import summary
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import h5py
import deepspeed
def parse_args():
    # 创建ArgumentParser对象
        parser = argparse.ArgumentParser(description="Training parameters for IQ signal processing with Transformer.")

        # 添加参数
        parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train the model (default: 100)')
        parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training (default: 128)')
        parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for optimizer (default: 1e-4)')
    
        parser.add_argument('--input_dim', type=int, default=2, help='Input dimension of the IQ data (default: 2)')
        
        parser.add_argument('--max_seq_length', type=int, default=128, help='Maximum sequence length for the input data (default: 128)')
        parser.add_argument('--num_workers', type=int, default=12, help='Num workers for data loader')
        parser.add_argument('--model_name', type=str, default="Our", help='Num workers for data loader')
        parser.add_argument('--task_name', type=str, default="wtc", help='The task name')
        parser.add_argument('--local_rank', type=int, default=0, help='local rank passed from distributed launcher')
        # 解析命令行参数
        args = parser.parse_args()

        return args
def process_data():
    with open("Data/RML2016.10a_dict.pkl", "rb",) as f:
        data = pickle.load(f,encoding="latin1")
    amc_label_dict = {}
    for k in data.keys():
        amc, snr = k
        if amc not in amc_label_dict:
            amc_label_dict[amc] = len(amc_label_dict)
    # 定义存储的字典
    train_dict = {"value": [], "label": []}
    test_dict = {"value": [], "label": [], "snr": []}

    # 遍历数据集
    for k, v in data.items():
        total_len = len(v)
        
        # 使用 sklearn 的 train_test_split 划分数据集
        train_indices, test_indices = train_test_split(np.arange(total_len), test_size=0.9, random_state=42)
        
        # 获取训练集和测试集
        train_values = v[train_indices]
        test_values = v[test_indices]
        
        # 获取对应的标签和 SNR
        train_labels = [amc_label_dict[k[0]]] * len(train_indices)
        test_labels = [amc_label_dict[k[0]]] * len(test_indices)
        test_snrs = [k[1]] * len(test_indices)
        
        # 将数据添加到字典中
        train_dict["value"].append(train_values)
        train_dict["label"].append(train_labels)
        
        test_dict["value"].append(test_values)
        test_dict["label"].append(test_labels)
        test_dict["snr"].append(test_snrs)

    # 合并所有的数据，生成最终的 numpy 数组
    train_dict["value"] = np.concatenate(train_dict["value"], axis=0)
    train_dict["value"] = np.transpose(train_dict["value"], (0, 2, 1))

    # ************************iq2ap*******************
    train_dict["value"] = iq2ap(train_dict["value"])
    # # #********************min-max-scaling********************
    train_dict["value"] = normalize(train_dict["value"])
    # train_dict["value"][:, :, 0] = (train_dict["value"][:, :, 0] - 0.04633322) / 0.020672457
    # train_dict["value"][:, :, 1] = (train_dict["value"][:, :, 1] - 0.4819608) / 0.28311247
    
    # print(train_dict["value"][:, :, 0].mean())
    # print(train_dict["value"][:, :, 0].std())
    # print(train_dict["value"][:, :, 1].mean())
    # print(train_dict["value"][:, :, 1].std())
################################0.16861732

    train_dict["label"] = np.concatenate(train_dict["label"], axis=0)
    test_dict["value"] = np.concatenate(test_dict["value"], axis=0)
    test_dict["value"] = np.transpose(test_dict["value"], (0, 2, 1))
    test_dict["value"] = iq2ap(test_dict["value"])
    test_dict["value"] = normalize(test_dict["value"])
    # test_dict["value"][:, :, 0] = (test_dict["value"][:, :, 0] - 0.04633322) / 0.020672457
    # test_dict["value"][:, :, 1] = (test_dict["value"][:, :, 1] - 0.4819608) / 0.28311247
    test_dict["label"] = np.concatenate(test_dict["label"], axis=0)
    test_dict["snr"] = np.concatenate(test_dict["snr"], axis=0)
    return train_dict, test_dict
def process_data_wtc():
    
    train_dict = {"value": [], "label": []}
    test_dict = {"value": [], "label": [], "snr": []}
    with h5py.File("Data/output_split.h5", "r") as f:
            sample = f["train_X"][:20000]  # 延迟加载样本
            label = np.squeeze(f["train_Y"][:20000]) # 延迟加载标签

    train_dict["value"] = sample  
    train_dict["label"] = label
    with h5py.File("Data/output_split.h5", "r") as f:
            sample = f["test_X"][:]  # 延迟加载样本
            label = np.squeeze(f["test_Y"][:])   # 延迟加载标签
            snr = f["test_Z"][:]
    test_dict["value"] = sample  
    test_dict["label"] = label
    test_dict["snr"] = snr
    
   
    return train_dict, test_dict
def load_h5():
    with h5py.File("Data/wtc.h5", "r") as f:
                sample = f["X"][:]  
                label = np.squeeze(f["Y"][:])  
                snr = f["Z"][:]
    length = len(sample)
    train_idx, test_idx = train_test_split(list(range(length)), test_size=0.1, random_state=42)
    train_dict = {"value": sample[train_idx],
                "label": label[train_idx]}
    test_dict = {"value": sample[test_idx],
                "label": label[test_idx],
                 "snr": snr[test_idx]}
    return train_dict, test_dict
def train(model, train_dataloader, val_dataloader, args):
    optimizer = optim.Adam((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    # model_engine, optimizer, _, _ = deepspeed.initialize(args=None, model=model, optimizer=optimizer, model_parameters=model.parameters(), config_params= "deepspeed_config.json")
    criterion = nn.CrossEntropyLoss() 
    early_stopping = EarlyStopping(patience=10, verbose=True, path=f'Checkpoint/{args.task_name}/{args.model_name}.pt', monitor="accuracy", pretrain=False)
    for epoch in range(args.epochs):
        model.train()
        train_loss = []
        train_acc = []
        for value, label in tqdm(train_dataloader):
            value, label = value.to(device), label.to(device)
            output = model(value)
            if args.model_name == "DAE":
                output, rec = output
                rec_loss = F.mse_loss(rec, value)
                original_loss = criterion(output, label)
                loss = 0.5 * original_loss  + 0.5 * rec_loss
            else:
                loss = criterion(output, label)
            train_loss.append(loss.item())
            predicted_label = np.argmax(output.detach().cpu().numpy(), axis=1)
            accuracy = accuracy_score(label.cpu().numpy(), predicted_label)
            train_acc.append(accuracy)
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # model_engine.backward(loss)
            # model_engine.step()
            # scheduler.step()
            # 输出当前epoch的损失
        print(f"Epoch [{epoch+1}/{args.epochs}], TrainLoss: {np.mean(train_loss):.7f}, TrainAcc: {np.mean(train_acc):.7f}")
        test_loss, test_acc = [], []
        model.eval()
        with torch.no_grad():
            for value, label, snr in tqdm(test_dataloader):
                value, label = value.to(device), label.to(device)
                output = model(value)
                if args.model_name == "DAE":
                    output, rec = output
                    rec_loss = F.mse_loss(rec, value)
                    original_loss = criterion(output, label)
                    loss = 0.5 * original_loss  + 0.5 * rec_loss
                else:
                    loss = criterion(output, label)
                
                test_loss.append(loss.item())
                predicted_label = np.argmax(output.detach().cpu().numpy(), axis=1)
                accuracy = accuracy_score(label.cpu().numpy(), predicted_label)
                test_acc.append(accuracy)
            print(f"Epoch [{epoch+1}/{args.epochs}], TestLoss: {np.mean(test_loss):.7f}, TestAcc: {np.mean(test_acc):.7f}")
            # print(np.mean(test_loss))
        # torch.save(model.state_dict(),"checkpoint/amc/resnet.pt")
            early_stopping(np.mean(test_acc), model)

                # 检查是否应该提前停止
            if early_stopping.early_stop:
                print("Early stopping")
                break

def test(model, test_dataloader, args):
    model.load_state_dict(torch.load(f'Checkpoint/{args.task_name}/{args.model_name}.pt'))    
    all_labels = []
    all_predictions = []
    model.eval()
    with torch.no_grad():
        for value, label, snr in tqdm(test_dataloader):
            value, label = value.to(device), label.to(device)
            output = model(value)
            if args.model_name == "DAE":
                output, rec = output
            predicted_label = np.argmax(output.detach().cpu().numpy(), axis=1)
            
            # 收集所有真实标签和预测值
            all_labels.extend(label.cpu().numpy())
            all_predictions.extend(predicted_label)

        # 转为 NumPy 数组
        all_labels = np.array(all_labels)
        all_predictions = np.array(all_predictions)


        oa = accuracy_score(all_labels, all_predictions)  # Overall Accuracy
        precision = precision_score(all_labels, all_predictions, average='weighted')  # 加权平均精确率
        recall = recall_score(all_labels, all_predictions, average='weighted')  # 加权平均召回率
        f1 = f1_score(all_labels, all_predictions, average='weighted')  # 加权平均 F1 分数
        # confusion = confusion_matrix(all_labels, all_predictions)
        # # labels=['QPSK','PAM4', 'AM-DSB','GFSK', 'QAM64', 'AM-SSB', 'QAM16', '8PSK', 'WBFM', 'BPSK', 'CPFSK']
        # print(confusion)

        print(f"Overall Accuracy: {oa:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        re = [f"{args.model_name} Overall Accuracy: {oa:.4f}", f"Precision: {precision:.4f}", f"Recall: {recall:.4f}", f"F1 Score: {f1:.4f}"]
    with open(f'Results/{args.task_name}/ov_results.csv', 'a', newline="") as f:
        writer = csv.writer(f)
        writer.writerow(re)
    
def test_per_snr(model, test_dataloader, args):
    """
    评估模型在测试集上的分 SNR 准确率和损失。
    """
     
    model.load_state_dict(torch.load(f'Checkpoint/{args.task_name}/{args.model_name}.pt'))
    model.eval()
    label = {}
    predict = {}
    results = {}
    
    with torch.no_grad():
        for data, labels, snrs in tqdm(test_dataloader):
            data, labels, snrs = data.to(device), labels.to(device), snrs.to(device)
            
            # 模型前向计算
            outputs = model(data)
            if args.model_name == "DAE":
                outputs, rec = outputs
            preds = outputs.argmax(dim=1)
            # 将每个样本分配到对应的 SNR 分组
            for i in range(len(snrs)):
                snr = snrs[i].item()
               
                
                if snr not in predict:
                    predict[snr] = []
                predict[snr].append(preds[i].item())
                if snr not in label:
                    label[snr] = []
                label[snr].append(labels[i].item())
                
                
    
    # 计算每个 SNR 的准确率和平均损失
    accs = [f"{args.model_name}"]
    for snr in sorted(label.keys()):
        results[snr] = {}
        results[snr]['accuracy'] = accuracy_score(label[snr], predict[snr])
        
        print(f'SNR: {snr}, Accuracy: {results[snr]["accuracy"]:.4f}')
        accs.append(results[snr]['accuracy'])
    with open(f'Results/{args.task_name}/snr_results.csv', 'a', newline="") as f:
        writer = csv.writer(f)
        writer.writerow(accs)
        
    return results

def confusion(model, test_dataloader, args):
    """
    评估模型在测试集上的分 SNR 准确率和损失。
    """
     
    model.load_state_dict(torch.load(f'Checkpoint/{args.task_name}/{args.model_name}.pt'))
    model.eval()
    label = {}
    predict = {}
    results = {}
    
    with torch.no_grad():
        for data, labels, snrs in tqdm(test_dataloader):
            data, labels, snrs = data.to(device), labels.to(device), snrs.to(device)
            
            # 模型前向计算
            outputs = model(data)
            if args.model_name == "DAE":
                outputs, rec = outputs
            preds = outputs.argmax(dim=1)
            # 将每个样本分配到对应的 SNR 分组
            for i in range(len(snrs)):
                snr = snrs[i].item()
               
                
                if snr not in predict:
                    predict[snr] = []
                predict[snr].append(preds[i].item())
                if snr not in label:
                    label[snr] = []
                label[snr].append(labels[i].item())
                
                
    
    # 计算每个 SNR 的准确率和平均损失
    
    for snr in sorted(label.keys()):
        cm = confusion_matrix(label[snr], predict[snr])
        with open(f'Confusion_matrix/{args.task_name}/'+args.model_name+'_'+str(snr)+'.csv', 'w', newline="") as f:
            writer = csv.writer(f)
            writer.writerows(cm)
            
        
        
    
    return results
def tsne(model, test_dataloader, args):
    model.load_state_dict(torch.load(f'Checkpoint/{args.task_name}/{args.model_name}.pt'))
    model.eval()
    label = {}
    predict = {}
    results = {}
    
    with torch.no_grad():
        for data, labels, snrs in tqdm(test_dataloader):
            data, labels, snrs = data.to(device), labels.to(device), snrs.to(device)
            
            # 模型前向计算
            outputs = model(data)
            if args.model_name == "DAE":
                outputs, rec = outputs
            for i in range(len(snrs)):
                snr = snrs[i].item()
               
                
                if snr not in predict:
                    predict[snr] = []
                predict[snr].append(np.expand_dims(outputs[i].detach().cpu().numpy(), axis=0))
                if snr not in label:
                    label[snr] = []
                label[snr].append(labels[i].item())
                
                
    
    # 计算每个 SNR 的准确率和平均损失
    
    for snr in sorted(label.keys()):
        
        if snr == 0:
            embeddings = np.concatenate(predict[snr], axis=0)
            labels = np.array(label[snr])
            # 使用 t-SNE 降维到 2D
            tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
            features_2d = tsne.fit_transform(embeddings)

            # 设置不同类别的颜色
            colors = plt.cm.get_cmap("tab10", 11)  # tab10 是一个颜色映射

            # 绘制 t-SNE 图
            plt.figure(figsize=(10, 8))
            classes=['QPSK','PAM4', 'AM-DSB','GFSK', 'QAM64', 'AM-SSB', 'QAM16', '8PSK', 'WBFM', 'BPSK', 'CPFSK']
            for class_id in range(11):
                indices = np.where(labels == class_id)
                plt.scatter(features_2d[indices, 0], features_2d[indices, 1],
                            label=classes[class_id], color=colors(class_id))

            # 添加图例和标题
            plt.legend()
            # plt.title("t-SNE Visualization of Features by Class")
            plt.xlabel("t-SNE Dimension 1")
            plt.ylabel("t-SNE Dimension 2")
            plt.grid(True)
            plt.savefig(f'Results/{args.task_name}/{args.model_name}_{snr}db_tsne.svg')


                   
if __name__ == '__main__':
    args = parse_args()
    device = torch.device(f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    print(f"Using device: {device}")
    #################################### Prepare data #################################### 
    # train_dict, test_dict = process_data()
    #***********************************load data from h5 ********************************
    # train_dict, test_dict = load_h5()
    train_dict, test_dict = process_data_wtc()
    train_dataset = AMCDataset(train_dict)
    test_dataset = AMCDataset(test_dict)
    val_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)
    ################################ Model ###################################################
    #model = AMC_Net(11, 128, 36).to(device)
    #model = MSNet(2, 3).to(device)
    #model = ResNet(2, 11).to(device)
    # model = VGG(2, 11).to(device)
    #model = CNN2(1, 11).to(device)
    # model = DAE((128, 2), 11).to(device)
    # model = CGDNN(1, 11).to(device)
    #model = MCNet(11).to(device)
    #model = Transformer().to(device)
    #model = GRU2(2, 128, 11).to(device)
   
    ################################ Our Model ############################################
    model = ConformerClassifier(2, 256, 4, 16, 512, 3, 129).to(device)# 64/128 up to 7 layers
    # model.encoder.load_state_dict(torch.load("Checkpoint/pretrain_model_full.pt"))
    # for layer in model.encoder.layers[:-6]:  # 假设 model.encoder.layers 是 ModuleList
    #     for param in layer.parameters():
    #         param.requires_grad = False
    ################################################################################
    # model = LLMClassifier(2, 256, 4, 16, 512, 3, 129).to(device)
    
    
    
    ####################################################
    
    summary(model, input_size=(256, 128, 2))
    train(model, train_dataloader, test_dataloader, args)
    # test(model, test_dataloader, args)
    test_per_snr(model, test_dataloader, args)
    # confusion(model, test_dataloader, args)
    # tsne(model, test_dataloader, args)


