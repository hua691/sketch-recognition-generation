"""
Sketch Recognition 主程序
"""

import os
import argparse
import torch
import numpy as np
from datetime import datetime

from dataset import create_dataloaders
from model import SketchRecognitionModel
from train import Trainer
from evaluate import Evaluator


def main():
    parser = argparse.ArgumentParser(description='Sketch Recognition')
    
    parser.add_argument('--mode', type=str, default='all',
                       choices=['train', 'evaluate', 'all'])
    parser.add_argument('--data_dir', type=str, default='D:/AAAAAAAAAAAAAproject/QuickDraw414k')
    parser.add_argument('--categories', type=str, nargs='+',
                       default=['cat', 'dog', 'fish', 'bird', 'apple', 'car', 'house', 'tree', 'flower', 'sun'])
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--backbone', type=str, default='resnet18')
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--max_samples', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("="*60)
    print("SKETCH RECOGNITION")
    print("="*60)
    print(f"Mode: {args.mode}")
    print(f"Device: {device}")
    print(f"Categories: {args.categories}")
    
    # 加载数据
    print("\nLoading data...")
    train_loader, val_loader, test_loader, categories = create_dataloaders(
        args.data_dir,
        categories=args.categories,
        batch_size=args.batch_size,
        max_samples_per_class=args.max_samples
    )
    
    if args.mode in ['train', 'all']:
        print("\n" + "="*60)
        print("TRAINING")
        print("="*60)
        
        model = SketchRecognitionModel(
            num_classes=len(categories),
            backbone=args.backbone,
            pretrained=True
        )
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        config = vars(args)
        config['categories'] = categories
        
        trainer = Trainer(model, train_loader, val_loader, config, device)
        best_acc = trainer.train(args.epochs, args.save_dir)
        print(f"\nBest validation accuracy: {best_acc:.2f}%")
    
    if args.mode in ['evaluate', 'all']:
        print("\n" + "="*60)
        print("EVALUATION")
        print("="*60)
        
        checkpoint_path = os.path.join(args.save_dir, 'best_recognition_model.pth')
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            model = SketchRecognitionModel(
                num_classes=len(categories),
                backbone=args.backbone
            )
            model.load_state_dict(checkpoint['model_state_dict'])
            
            evaluator = Evaluator(model, test_loader, categories, device)
            evaluator.full_evaluation(os.path.join(args.save_dir, 'recognition_results'))
        else:
            print(f"No checkpoint found at {checkpoint_path}")
    
    print("\n" + "="*60)
    print("COMPLETED")
    print("="*60)


if __name__ == '__main__':
    main()
