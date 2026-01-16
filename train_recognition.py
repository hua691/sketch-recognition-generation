"""
Training script for Sketch Recognition (Task 2A)
"""

import os
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import matplotlib.pyplot as plt

from data.quickdraw_dataset import QuickDrawDataset, create_dataloaders
from models.sketch_recognition import get_recognition_model


class Trainer:
    def __init__(self, model, train_loader, val_loader, config, device='cuda'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=config['lr'],
            weight_decay=config['weight_decay']
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer, 
            T_max=config['epochs'],
            eta_min=1e-7
        )
        
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        self.best_val_acc = 0
    
    def train_epoch(self):
        self.model.train()
        total_loss, correct, total = 0, 0, 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch in pbar:
            if len(batch) == 2:
                data, labels = batch
                data = data.to(self.device)
            else:
                seq, img, labels = batch
                seq, img = seq.to(self.device), img.to(self.device)
                data = (seq, img)
            
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            if isinstance(data, tuple):
                outputs = self.model(*data)
            else:
                outputs = self.model(data)
            
            loss = self.criterion(outputs, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item() * labels.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'loss': loss.item(), 'acc': 100.*correct/total})
        
        return total_loss / total, 100. * correct / total
    
    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss, correct, total = 0, 0, 0
        
        for batch in self.val_loader:
            if len(batch) == 2:
                data, labels = batch
                data = data.to(self.device)
            else:
                seq, img, labels = batch
                seq, img = seq.to(self.device), img.to(self.device)
                data = (seq, img)
            
            labels = labels.to(self.device)
            
            if isinstance(data, tuple):
                outputs = self.model(*data)
            else:
                outputs = self.model(data)
            
            loss = self.criterion(outputs, labels)
            
            total_loss += loss.item() * labels.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        return total_loss / total, 100. * correct / total
    
    def train(self, epochs, save_dir='checkpoints'):
        os.makedirs(save_dir, exist_ok=True)
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()
            
            self.scheduler.step()
            
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'val_acc': val_acc,
                    'config': self.config
                }, os.path.join(save_dir, 'best_recognition_model.pth'))
                print(f"Saved best model with val_acc: {val_acc:.2f}%")
        
        # Save history
        with open(os.path.join(save_dir, 'recognition_history.json'), 'w') as f:
            json.dump(self.history, f)
        
        self.plot_history(save_dir)
        
        return self.history
    
    def plot_history(self, save_dir):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        axes[0].plot(self.history['train_loss'], label='Train')
        axes[0].plot(self.history['val_loss'], label='Val')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].set_title('Loss Curve')
        
        axes[1].plot(self.history['train_acc'], label='Train')
        axes[1].plot(self.history['val_acc'], label='Val')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].legend()
        axes[1].set_title('Accuracy Curve')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'recognition_curves.png'), dpi=150)
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train Sketch Recognition Model')
    parser.add_argument('--data_dir', type=str, default='./quickdraw_data')
    parser.add_argument('--categories', type=str, nargs='+', 
                       default=['cat', 'dog', 'car', 'house', 'tree', 
                               'flower', 'bird', 'fish', 'apple', 'sun'])
    parser.add_argument('--model', type=str, default='transformer',
                       choices=['cnn', 'rnn', 'transformer', 'hybrid'])
    parser.add_argument('--representation', type=str, default='stroke3',
                       choices=['stroke3', 'stroke5', 'image', 'both'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--max_seq_len', type=int, default=200)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create data loaders
    train_loader, val_loader, test_loader, categories = create_dataloaders(
        args.data_dir, args.categories, args.batch_size,
        args.representation, args.max_seq_len, args.img_size
    )
    
    num_classes = len(categories)
    print(f"Number of classes: {num_classes}")
    print(f"Categories: {categories}")
    
    # Create model
    if args.representation == 'image':
        model = get_recognition_model('cnn', num_classes)
    elif args.representation in ['stroke3', 'stroke5']:
        input_dim = 3 if args.representation == 'stroke3' else 5
        model = get_recognition_model(args.model, num_classes, input_dim=input_dim)
    else:  # both
        model = get_recognition_model('hybrid', num_classes)
    
    # Config
    config = vars(args)
    
    # Train
    trainer = Trainer(model, train_loader, val_loader, config, device)
    trainer.train(args.epochs, args.save_dir)
    
    print(f"\nBest validation accuracy: {trainer.best_val_acc:.2f}%")


if __name__ == '__main__':
    main()
