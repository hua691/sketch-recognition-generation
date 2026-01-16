"""
Sketch Recognition 训练脚本
快速训练版本
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import json
import matplotlib.pyplot as plt

from dataset import create_dataloaders
from model import SketchRecognitionModel


class Trainer:
    def __init__(self, model, train_loader, val_loader, config, device='cpu'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.get('lr', 1e-3),
            weight_decay=config.get('weight_decay', 1e-4)
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.get('epochs', 20),
            eta_min=1e-6
        )
        
        self.best_acc = 0
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})
        
        self.scheduler.step()
        return total_loss / len(self.train_loader), 100. * correct / total
    
    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        for images, labels in self.val_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        return total_loss / len(self.val_loader), 100. * correct / total
    
    def train(self, epochs, save_dir='checkpoints'):
        os.makedirs(save_dir, exist_ok=True)
        
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()
            
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            print(f'Epoch {epoch+1}/{epochs}: Train Loss={train_loss:.4f}, Acc={train_acc:.2f}% | Val Loss={val_loss:.4f}, Acc={val_acc:.2f}%')
            
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'val_acc': val_acc,
                    'config': self.config
                }, os.path.join(save_dir, 'best_recognition_model.pth'))
                print(f'  Saved best model (acc: {val_acc:.2f}%)')
        
        # 保存历史
        with open(os.path.join(save_dir, 'recognition_history.json'), 'w') as f:
            json.dump(self.history, f)
        
        self.plot_curves(save_dir)
        return self.best_acc
    
    def plot_curves(self, save_dir):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        
        axes[0].plot(self.history['train_loss'], label='Train')
        axes[0].plot(self.history['val_loss'], label='Val')
        axes[0].set_title('Loss')
        axes[0].legend()
        
        axes[1].plot(self.history['train_acc'], label='Train')
        axes[1].plot(self.history['val_acc'], label='Val')
        axes[1].set_title('Accuracy (%)')
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'recognition_curves.png'), dpi=100)
        plt.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='D:/AAAAAAAAAAAAAproject/QuickDraw414k')
    parser.add_argument('--categories', type=str, nargs='+', 
                       default=['cat', 'dog', 'fish', 'bird', 'apple', 'car', 'house', 'tree', 'flower', 'sun'])
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--backbone', type=str, default='resnet18')
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--max_samples', type=int, default=500, help='Max samples per class for fast training')
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # 加载数据
    print("Loading data...")
    train_loader, val_loader, test_loader, categories = create_dataloaders(
        args.data_dir,
        categories=args.categories,
        batch_size=args.batch_size,
        max_samples_per_class=args.max_samples
    )
    
    print(f"Categories ({len(categories)}): {categories}")
    
    # 创建模型
    model = SketchRecognitionModel(
        num_classes=len(categories),
        backbone=args.backbone,
        pretrained=True
    )
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 训练
    config = vars(args)
    config['categories'] = categories
    
    trainer = Trainer(model, train_loader, val_loader, config, device)
    best_acc = trainer.train(args.epochs, args.save_dir)
    
    print(f"\nBest validation accuracy: {best_acc:.2f}%")


if __name__ == '__main__':
    main()
