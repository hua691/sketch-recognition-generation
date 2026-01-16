"""
Training script for Sketch Generation (Task 2B)
Supports VAE and Diffusion models
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
from models.sketch_generation import SketchRNNVAE, ConditionalSketchVAE, sketch_rnn_loss
from models.sketch_diffusion import SketchKnitter


class VAETrainer:
    """Trainer for Sketch-RNN VAE models."""
    
    def __init__(self, model, train_loader, val_loader, config, device='cuda'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        self.optimizer = optim.Adam(model.parameters(), lr=config['lr'])
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=config['epochs'])
        
        self.history = {'train_loss': [], 'val_loss': [], 'kl_loss': [], 'recon_loss': []}
        self.best_val_loss = float('inf')
        
        # KL annealing
        self.kl_weight = 0.0
        self.kl_weight_max = config.get('kl_weight', 0.5)
        self.kl_anneal_epochs = config.get('kl_anneal_epochs', 20)
    
    def train_epoch(self, epoch):
        self.model.train()
        total_loss, total_kl, total_recon = 0, 0, 0
        n_batches = 0
        
        # Update KL weight
        self.kl_weight = min(self.kl_weight_max, 
                           self.kl_weight_max * epoch / self.kl_anneal_epochs)
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch in pbar:
            if len(batch) == 2:
                data, labels = batch
            else:
                data, _, labels = batch
            
            data = data.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            if hasattr(self.model, 'class_embed'):
                output, mu, logvar = self.model(data, labels)
            else:
                output, mu, logvar, z = self.model(data)
            
            loss, recon_xy, recon_pen, kl = sketch_rnn_loss(
                output, data, mu, logvar, 
                self.model.decoder, self.kl_weight
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            total_kl += kl.item()
            total_recon += (recon_xy.item() + recon_pen.item())
            n_batches += 1
            
            pbar.set_postfix({
                'loss': loss.item(), 
                'kl': kl.item(),
                'kl_w': self.kl_weight
            })
        
        return total_loss/n_batches, total_kl/n_batches, total_recon/n_batches
    
    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0
        n_batches = 0
        
        for batch in self.val_loader:
            if len(batch) == 2:
                data, labels = batch
            else:
                data, _, labels = batch
            
            data = data.to(self.device)
            labels = labels.to(self.device)
            
            if hasattr(self.model, 'class_embed'):
                output, mu, logvar = self.model(data, labels)
            else:
                output, mu, logvar, z = self.model(data)
            
            loss, _, _, _ = sketch_rnn_loss(
                output, data, mu, logvar,
                self.model.decoder, self.kl_weight
            )
            
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / n_batches
    
    def train(self, epochs, save_dir='checkpoints'):
        os.makedirs(save_dir, exist_ok=True)
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            train_loss, kl_loss, recon_loss = self.train_epoch(epoch)
            val_loss = self.validate()
            
            self.scheduler.step()
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['kl_loss'].append(kl_loss)
            self.history['recon_loss'].append(recon_loss)
            
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            print(f"KL Loss: {kl_loss:.4f}, Recon Loss: {recon_loss:.4f}")
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'val_loss': val_loss,
                    'config': self.config
                }, os.path.join(save_dir, 'best_vae_model.pth'))
                print(f"Saved best model")
            
            # Generate samples periodically
            if (epoch + 1) % 10 == 0:
                self.generate_samples(save_dir, epoch + 1)
        
        with open(os.path.join(save_dir, 'vae_history.json'), 'w') as f:
            json.dump(self.history, f)
        
        return self.history
    
    @torch.no_grad()
    def generate_samples(self, save_dir, epoch):
        """Generate and save sample sketches."""
        self.model.eval()
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        
        for i in range(8):
            ax = axes[i // 4, i % 4]
            
            # Generate
            z = torch.randn(1, self.model.latent_dim).to(self.device)
            
            if hasattr(self.model, 'class_embed'):
                label = torch.tensor([i % self.model.num_classes]).to(self.device)
                sketch = self.model.generate(label, z, max_len=100, device=self.device)
            else:
                sketch = self.model.generate(z, max_len=100, device=self.device)
            
            # Plot
            self.plot_sketch(sketch.cpu().numpy(), ax)
            ax.set_title(f'Sample {i+1}')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'samples_epoch_{epoch}.png'), dpi=150)
        plt.close()
    
    def plot_sketch(self, stroke5, ax):
        """Plot stroke-5 format sketch."""
        # Convert to absolute coordinates
        x, y = 0, 0
        points = [(x, y)]
        pen_down = True
        
        for point in stroke5:
            dx, dy = point[0], point[1]
            x += dx
            y += dy
            
            if pen_down:
                points.append((x, y))
            else:
                if len(points) > 1:
                    xs, ys = zip(*points)
                    ax.plot(xs, ys, 'k-', linewidth=1)
                points = [(x, y)]
            
            # Check pen state
            if len(point) >= 5:
                if point[3] > 0.5:  # pen up
                    pen_down = False
                elif point[4] > 0.5:  # end
                    break
                else:
                    pen_down = True
            elif point[2] > 0.5:  # stroke-3 pen up
                pen_down = False
        
        if len(points) > 1:
            xs, ys = zip(*points)
            ax.plot(xs, ys, 'k-', linewidth=1)
        
        ax.set_aspect('equal')
        ax.axis('off')


class DiffusionTrainer:
    """Trainer for SketchKnitter diffusion model."""
    
    def __init__(self, model, train_loader, val_loader, config, device='cuda'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        self.optimizer = optim.AdamW(model.parameters(), lr=config['lr'])
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=config['epochs'])
        
        self.history = {'train_loss': [], 'val_loss': []}
        self.best_val_loss = float('inf')
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        n_batches = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch in pbar:
            if len(batch) == 2:
                data, labels = batch
            else:
                data, _, labels = batch
            
            data = data.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            loss = self.model(data, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
            
            pbar.set_postfix({'loss': loss.item()})
        
        return total_loss / n_batches
    
    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0
        n_batches = 0
        
        for batch in self.val_loader:
            if len(batch) == 2:
                data, labels = batch
            else:
                data, _, labels = batch
            
            data = data.to(self.device)
            labels = labels.to(self.device)
            
            loss = self.model(data, labels)
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / n_batches
    
    def train(self, epochs, save_dir='checkpoints'):
        os.makedirs(save_dir, exist_ok=True)
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            train_loss = self.train_epoch()
            val_loss = self.validate()
            
            self.scheduler.step()
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'val_loss': val_loss,
                    'config': self.config
                }, os.path.join(save_dir, 'best_diffusion_model.pth'))
                print(f"Saved best model")
            
            if (epoch + 1) % 10 == 0:
                self.generate_samples(save_dir, epoch + 1)
        
        return self.history
    
    @torch.no_grad()
    def generate_samples(self, save_dir, epoch):
        self.model.eval()
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        
        for i in range(8):
            ax = axes[i // 4, i % 4]
            label = torch.tensor([i % self.model.num_classes]).to(self.device)
            
            sketch = self.model.generate(
                label, batch_size=1, seq_len=100,
                device=self.device, use_ddim=True, num_inference_steps=20
            )
            
            # Plot
            self.plot_sketch(sketch[0].cpu().numpy(), ax)
            ax.set_title(f'Class {i % self.model.num_classes}')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'diffusion_samples_epoch_{epoch}.png'), dpi=150)
        plt.close()
    
    def plot_sketch(self, stroke5, ax):
        x, y = 0, 0
        points = [(x, y)]
        
        for point in stroke5:
            dx, dy = point[0], point[1]
            x += dx * 10  # Scale
            y += dy * 10
            points.append((x, y))
            
            if len(point) >= 5 and point[4] > 0.5:
                break
        
        if len(points) > 1:
            xs, ys = zip(*points)
            ax.plot(xs, ys, 'k-', linewidth=1)
        
        ax.set_aspect('equal')
        ax.axis('off')


def main():
    parser = argparse.ArgumentParser(description='Train Sketch Generation Model')
    parser.add_argument('--data_dir', type=str, default='./quickdraw_data')
    parser.add_argument('--categories', type=str, nargs='+',
                       default=['cat', 'dog', 'car', 'house', 'tree'])
    parser.add_argument('--model', type=str, default='vae',
                       choices=['vae', 'cvae', 'diffusion'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--latent_dim', type=int, default=128)
    parser.add_argument('--max_seq_len', type=int, default=200)
    parser.add_argument('--kl_weight', type=float, default=0.5)
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create data loaders (stroke-5 format for generation)
    train_loader, val_loader, test_loader, categories = create_dataloaders(
        args.data_dir, args.categories, args.batch_size,
        'stroke5', args.max_seq_len
    )
    
    num_classes = len(categories)
    print(f"Number of classes: {num_classes}")
    
    config = vars(args)
    config['num_classes'] = num_classes
    
    # Create model
    if args.model == 'vae':
        model = SketchRNNVAE(latent_dim=args.latent_dim)
        trainer = VAETrainer(model, train_loader, val_loader, config, device)
    elif args.model == 'cvae':
        model = ConditionalSketchVAE(num_classes, latent_dim=args.latent_dim)
        trainer = VAETrainer(model, train_loader, val_loader, config, device)
    else:  # diffusion
        model = SketchKnitter(num_classes, max_len=args.max_seq_len)
        trainer = DiffusionTrainer(model, train_loader, val_loader, config, device)
    
    # Train
    trainer.train(args.epochs, args.save_dir)
    
    print(f"\nTraining completed!")


if __name__ == '__main__':
    main()
