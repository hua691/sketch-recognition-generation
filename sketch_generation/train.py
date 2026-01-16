"""
Sketch Generation 训练脚本
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from tqdm import tqdm
import json
import matplotlib.pyplot as plt

from dataset import create_dataloaders
from model import SketchVAE, compute_loss


class Trainer:
    def __init__(self, model, train_loader, val_loader, config, device='cpu'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config.get('lr', 1e-3),
            weight_decay=config.get('weight_decay', 1e-5)
        )
        
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.get('epochs', 100),
            eta_min=1e-6
        )
        
        self.kl_weight = config.get('kl_weight', 0.5)
        self.kl_weight_start = config.get('kl_weight_start', 0.01)
        self.kl_anneal_epochs = config.get('kl_anneal_epochs', 20)
        
        self.best_val_loss = float('inf')
        self.history = {'train_loss': [], 'val_loss': [], 'kl_loss': [], 'recon_loss': []}
    
    def get_kl_weight(self, epoch):
        """KL权重退火"""
        if epoch < self.kl_anneal_epochs:
            return self.kl_weight_start + (self.kl_weight - self.kl_weight_start) * epoch / self.kl_anneal_epochs
        return self.kl_weight
    
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        total_recon = 0
        total_kl = 0
        
        kl_w = self.get_kl_weight(epoch)
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}')
        for batch in pbar:
            sketch = batch['sketch'].to(self.device)
            mask = batch['mask'].to(self.device)
            seq_lens = batch['seq_len']
            
            # 前向传播
            outputs, mu, logvar = self.model(sketch, seq_lens)
            
            # 计算损失 (目标是sketch[:, 1:])
            targets = sketch[:, 1:, :]
            target_mask = mask[:, 1:]
            
            loss, recon_loss, kl_loss = compute_loss(
                outputs, targets, mu, logvar, target_mask,
                kl_weight=kl_w, num_mixtures=self.model.num_mixtures
            )
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'recon': f'{recon_loss.item():.4f}',
                'kl': f'{kl_loss.item():.4f}'
            })
        
        self.scheduler.step()
        
        n = len(self.train_loader)
        return total_loss/n, total_recon/n, total_kl/n
    
    @torch.no_grad()
    def validate(self, epoch):
        self.model.eval()
        total_loss = 0
        total_recon = 0
        total_kl = 0
        
        kl_w = self.get_kl_weight(epoch)
        
        for batch in self.val_loader:
            sketch = batch['sketch'].to(self.device)
            mask = batch['mask'].to(self.device)
            seq_lens = batch['seq_len']
            
            outputs, mu, logvar = self.model(sketch, seq_lens)
            targets = sketch[:, 1:, :]
            target_mask = mask[:, 1:]
            
            loss, recon_loss, kl_loss = compute_loss(
                outputs, targets, mu, logvar, target_mask,
                kl_weight=kl_w, num_mixtures=self.model.num_mixtures
            )
            
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
        
        n = len(self.val_loader)
        return total_loss/n, total_recon/n, total_kl/n
    
    def train(self, epochs, save_dir='checkpoints'):
        os.makedirs(save_dir, exist_ok=True)
        
        for epoch in range(epochs):
            train_loss, train_recon, train_kl = self.train_epoch(epoch)
            val_loss, val_recon, val_kl = self.validate(epoch)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['recon_loss'].append(val_recon)
            self.history['kl_loss'].append(val_kl)
            
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'  Train Loss: {train_loss:.4f} (Recon: {train_recon:.4f}, KL: {train_kl:.4f})')
            print(f'  Val Loss: {val_loss:.4f} (Recon: {val_recon:.4f}, KL: {val_kl:.4f})')
            
            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'config': self.config
                }, os.path.join(save_dir, 'best_generation_model.pth'))
                print(f'  Saved best model (val_loss: {val_loss:.4f})')
            
            # 每10个epoch生成样本
            if (epoch + 1) % 10 == 0:
                self.generate_samples(save_dir, epoch)
        
        # 保存训练历史
        with open(os.path.join(save_dir, 'generation_history.json'), 'w') as f:
            json.dump(self.history, f)
        
        self.plot_curves(save_dir)
        
        return self.best_val_loss
    
    @torch.no_grad()
    def generate_samples(self, save_dir, epoch, num_samples=5):
        """生成并保存样本"""
        self.model.eval()
        
        samples = self.model.generate(
            num_samples=num_samples,
            max_len=150,
            temperature=0.4,
            device=self.device
        )
        
        # 可视化
        fig, axes = plt.subplots(1, num_samples, figsize=(3*num_samples, 3))
        if num_samples == 1:
            axes = [axes]
        
        for i, ax in enumerate(axes):
            sketch = samples[i].cpu().numpy()
            self.draw_sketch(ax, sketch)
            ax.set_title(f'Sample {i+1}')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'generated_epoch_{epoch+1}.png'), dpi=100)
        plt.close()
    
    def draw_sketch(self, ax, sketch):
        """绘制草图"""
        x, y = 0, 0
        lines_x, lines_y = [x], [y]
        
        for point in sketch:
            dx, dy, pen = point
            x += dx
            y += dy
            
            if pen < 0.5:  # pen down
                lines_x.append(x)
                lines_y.append(y)
            else:  # pen up
                if len(lines_x) > 1:
                    ax.plot(lines_x, lines_y, 'k-', linewidth=1)
                lines_x, lines_y = [x], [y]
        
        if len(lines_x) > 1:
            ax.plot(lines_x, lines_y, 'k-', linewidth=1)
        
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.axis('off')
    
    def plot_curves(self, save_dir):
        """绘制训练曲线"""
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        axes[0].plot(self.history['train_loss'], label='Train')
        axes[0].plot(self.history['val_loss'], label='Val')
        axes[0].set_title('Total Loss')
        axes[0].legend()
        
        axes[1].plot(self.history['recon_loss'])
        axes[1].set_title('Reconstruction Loss')
        
        axes[2].plot(self.history['kl_loss'])
        axes[2].set_title('KL Loss')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'generation_training_curves.png'), dpi=100)
        plt.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Sketch Generation Training')
    parser.add_argument('--data_dir', type=str, default='../sketch_generation_data')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--latent_dim', type=int, default=128)
    parser.add_argument('--enc_hidden', type=int, default=256)
    parser.add_argument('--dec_hidden', type=int, default=512)
    parser.add_argument('--max_seq_len', type=int, default=200)
    parser.add_argument('--kl_weight', type=float, default=0.5)
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--categories', type=str, nargs='+', default=None)
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # 加载数据
    print("Loading data...")
    train_loader, val_loader, test_loader, categories = create_dataloaders(
        args.data_dir,
        categories=args.categories,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len
    )
    
    print(f"Categories: {categories}")
    print(f"Train batches: {len(train_loader)}")
    
    # 创建模型
    model = SketchVAE(
        input_dim=3,
        enc_hidden=args.enc_hidden,
        dec_hidden=args.dec_hidden,
        latent_dim=args.latent_dim,
        num_mixtures=20
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 训练配置
    config = vars(args)
    config['categories'] = categories
    
    # 训练
    trainer = Trainer(model, train_loader, val_loader, config, device)
    best_loss = trainer.train(args.epochs, args.save_dir)
    
    print(f"\nTraining completed! Best val loss: {best_loss:.4f}")


if __name__ == '__main__':
    main()
