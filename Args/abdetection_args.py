import argparse

def parse_args():
    # 创建ArgumentParser对象
    parser = argparse.ArgumentParser(description="Training parameters for IQ signal processing with Transformer.")

    # 添加参数
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train the model (default: 100)')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training (default: 128)')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for optimizer (default: 1e-4)')
   
    parser.add_argument('--input_dim', type=int, default=2, help='Input dimension of the IQ data (default: 2)')
    
    parser.add_argument('--max_seq_length', type=int, default=128, help='Maximum sequence length for the input data (default: 128)')
    parser.add_argument('--num_workers', type=int, default=12, help='Num workers for data loader')
    
    # 解析命令行参数
    args = parser.parse_args()

    return args