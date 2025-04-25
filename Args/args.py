import argparse

def parse_args():
    # 创建ArgumentParser对象
    parser = argparse.ArgumentParser(description="Training parameters for IQ signal processing with Transformer.")

    # 添加参数
    parser.add_argument('--epochs', type=int, default=15, help='Number of epochs to train the model (default: 100)')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training (default: 128)')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for optimizer (default: 1e-4)')
    parser.add_argument('--peak_lr', type=float, default=4e-4, help='Peak learning rate for pretraining (default: 1e-4)')
    parser.add_argument('--input_dim', type=int, default=2, help='Input dimension of the IQ data (default: 2)')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension of the Transformer (default: 768)')
    parser.add_argument('--max_seq_length', type=int, default=128, help='Maximum sequence length for the input data (default: 128)')
    parser.add_argument('--mask_ratio', type=float, default=0.3, help='Ratio of input data to be masked (default: 0.9)')
    parser.add_argument('--nhead', type=int, default=4, help='Number of Transformer head')
    parser.add_argument('--dim_feedforward', type=int, default=512, help='Dim of feedforward layer')
    parser.add_argument('--num_layers', type=int, default=16, help='Number of Transformer layers')
    parser.add_argument('--warmup_steps', type=int, default=24000, help='Number of warmup steps')
    parser.add_argument('--total_steps', type=int, default=1707585, help='Number of total steps')
    parser.add_argument('--pretrain_data_path', type=str, default="Data/processed_2018_.h5", help='train data path')
    parser.add_argument('--pretrain_save_path', type=str, default="Checkpoint/pretrain_model_full_.pt", help='pretrain save path')
    parser.add_argument('--num_workers', type=int, default=12, help='Num workers for data loader')
    parser.add_argument('--noise_std', type=float, default=0.001, help='Noise Std')
    parser.add_argument('--local_rank', type=int, help='local rank passed from distributed launcher')
    # 解析命令行参数
    args = parser.parse_args()

    return args

