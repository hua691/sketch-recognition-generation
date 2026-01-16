"""
Sketch Generation 主程序
支持训练、评估和生成
"""

import os
import argparse
import torch
import numpy as np
from datetime import datetime

from dataset import create_dataloaders
from model import SketchVAE
from train import Trainer
from evaluate import GenerationEvaluator


def main():
    parser = argparse.ArgumentParser(description='Sketch Generation (Sketch-RNN VAE)')
    
    # 模式
    parser.add_argument('--mode', type=str, default='all',
                       choices=['train', 'evaluate', 'generate', 'all'],
                       help='Running mode')
    
    # 数据
    parser.add_argument('--data_dir', type=str, default='../sketch_generation_data',
                       help='Data directory with .npz files')
    parser.add_argument('--categories', type=str, nargs='+', 
                       default=['cat', 'apple', 'fish', 'pig', 'butterfly'],
                       help='Categories to use (5-10 recommended)')
    
    # 模型
    parser.add_argument('--latent_dim', type=int, default=128,
                       help='Latent dimension')
    parser.add_argument('--enc_hidden', type=int, default=256,
                       help='Encoder hidden dimension')
    parser.add_argument('--dec_hidden', type=int, default=512,
                       help='Decoder hidden dimension')
    parser.add_argument('--num_mixtures', type=int, default=20,
                       help='Number of GMM mixtures')
    
    # 训练
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--max_seq_len', type=int, default=200,
                       help='Maximum sequence length')
    parser.add_argument('--kl_weight', type=float, default=0.5,
                       help='KL divergence weight')
    parser.add_argument('--max_samples', type=int, default=2000,
                       help='Max samples per class for fast training')
    
    # 其他
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                       help='Save directory')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("="*60)
    print("SKETCH GENERATION - Sketch-RNN VAE")
    print("="*60)
    print(f"Mode: {args.mode}")
    print(f"Categories: {args.categories}")
    print(f"Device: {device}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 加载数据
    print("\nLoading data...")
    train_loader, val_loader, test_loader, categories = create_dataloaders(
        args.data_dir,
        categories=args.categories,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        max_samples_per_class=args.max_samples
    )
    
    print(f"Loaded categories: {categories}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    if args.mode in ['train', 'all']:
        print("\n" + "="*60)
        print("TRAINING")
        print("="*60)
        
        # 创建模型
        model = SketchVAE(
            input_dim=3,
            enc_hidden=args.enc_hidden,
            dec_hidden=args.dec_hidden,
            latent_dim=args.latent_dim,
            num_mixtures=args.num_mixtures
        )
        
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # 训练配置
        config = vars(args)
        config['categories'] = categories
        
        # 训练
        trainer = Trainer(model, train_loader, val_loader, config, device)
        best_loss = trainer.train(args.epochs, args.save_dir)
        
        print(f"\nBest validation loss: {best_loss:.4f}")
    
    if args.mode in ['evaluate', 'all']:
        print("\n" + "="*60)
        print("EVALUATION")
        print("="*60)
        
        checkpoint_path = os.path.join(args.save_dir, 'best_generation_model.pth')
        
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            model = SketchVAE(
                input_dim=3,
                enc_hidden=args.enc_hidden,
                dec_hidden=args.dec_hidden,
                latent_dim=args.latent_dim,
                num_mixtures=args.num_mixtures
            )
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from epoch {checkpoint['epoch']}")
            
            # 评估
            evaluator = GenerationEvaluator(model, test_loader, categories, device)
            results = evaluator.full_evaluation(os.path.join(args.save_dir, 'generation_results'))
        else:
            print(f"No checkpoint found at {checkpoint_path}")
    
    if args.mode == 'generate':
        print("\n" + "="*60)
        print("GENERATION")
        print("="*60)
        
        checkpoint_path = os.path.join(args.save_dir, 'best_generation_model.pth')
        
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            model = SketchVAE(
                input_dim=3,
                enc_hidden=args.enc_hidden,
                dec_hidden=args.dec_hidden,
                latent_dim=args.latent_dim,
                num_mixtures=args.num_mixtures
            )
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            model.eval()
            
            # 生成样本
            import matplotlib.pyplot as plt
            
            samples = model.generate(num_samples=10, max_len=150, temperature=0.4, device=device)
            
            fig, axes = plt.subplots(2, 5, figsize=(15, 6))
            axes = axes.flatten()
            
            for i, ax in enumerate(axes):
                sketch = samples[i].detach().cpu().numpy()
                x, y = 0, 0
                lines_x, lines_y = [x], [y]
                
                for point in sketch:
                    dx, dy, pen = point
                    x += dx
                    y += dy
                    
                    if pen < 0.5:
                        lines_x.append(x)
                        lines_y.append(y)
                    else:
                        if len(lines_x) > 1:
                            ax.plot(lines_x, lines_y, 'k-', linewidth=1.5)
                        lines_x, lines_y = [x], [y]
                
                if len(lines_x) > 1:
                    ax.plot(lines_x, lines_y, 'k-', linewidth=1.5)
                
                ax.set_aspect('equal')
                ax.invert_yaxis()
                ax.axis('off')
                ax.set_title(f'Sample {i+1}')
            
            plt.tight_layout()
            save_path = os.path.join(args.save_dir, 'generated_samples.png')
            plt.savefig(save_path, dpi=150)
            plt.close()
            print(f"Generated samples saved to {save_path}")
        else:
            print(f"No checkpoint found at {checkpoint_path}")
    
    print("\n" + "="*60)
    print("COMPLETED")
    print("="*60)


if __name__ == '__main__':
    main()
