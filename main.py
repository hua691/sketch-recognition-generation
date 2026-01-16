"""
Sketch Recognition - Task 2A
QuickDraw-414k Dataset Classification
"""

import os
import argparse
import torch
import numpy as np
from datetime import datetime

from data.quickdraw_dataset import QuickDrawDataset, create_dataloaders
from models.sketch_recognition import get_recognition_model
from train_recognition import Trainer
from evaluate_sketch import RecognitionEvaluator


def main():
    parser = argparse.ArgumentParser(description='Sketch Recognition (Task 2A)')
    
    # Mode
    parser.add_argument('--mode', type=str, default='all',
                       choices=['train', 'evaluate', 'all'],
                       help='Running mode')
    
    # Data
    parser.add_argument('--data_dir', type=str, default='./quickdraw_data/QuickDraw414k',
                       help='QuickDraw dataset directory')
    parser.add_argument('--categories', type=str, nargs='+',
                       default=['apple', 'banana', 'airplane', 'alarm_clock', 'ambulance'],
                       help='Categories to use (5-10 recommended)')
    
    # Model
    parser.add_argument('--model', type=str, default='cnn',
                       choices=['cnn', 'rnn', 'transformer'],
                       help='Model architecture')
    parser.add_argument('--representation', type=str, default='image',
                       choices=['stroke3', 'image'],
                       help='Data representation')
    
    # Training
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    
    # Other
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                       help='Save directory')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("="*60)
    print("SKETCH RECOGNITION - Task 2A")
    print("="*60)
    print(f"Mode: {args.mode}")
    print(f"Model: {args.model}")
    print(f"Categories: {args.categories}")
    print(f"Device: {device}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create data loaders
    train_loader, val_loader, test_loader, categories = create_dataloaders(
        args.data_dir, args.categories, args.batch_size, img_size=224
    )
    
    num_classes = len(categories)
    print(f"\nNumber of classes: {num_classes}")
    
    if args.mode in ['train', 'all']:
        print("\n" + "="*60)
        print("TRAINING")
        print("="*60)
        
        # Create model
        if args.representation == 'image':
            model = get_recognition_model('cnn', num_classes, pretrained=True)
        else:
            model = get_recognition_model(args.model, num_classes, input_dim=3)
        
        # Config
        config = vars(args)
        config['num_classes'] = num_classes
        
        # Train
        trainer = Trainer(model, train_loader, val_loader, config, device)
        trainer.train(args.epochs, args.save_dir)
        
        print(f"\nBest validation accuracy: {trainer.best_val_acc:.2f}%")
    
    if args.mode in ['evaluate', 'all']:
        print("\n" + "="*60)
        print("EVALUATION")
        print("="*60)
        
        # Load best model
        checkpoint_path = os.path.join(args.save_dir, 'best_recognition_model.pth')
        
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            if args.representation == 'image':
                model = get_recognition_model('cnn', num_classes, pretrained=False)
            else:
                model = get_recognition_model(args.model, num_classes, input_dim=3)
            
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from epoch {checkpoint['epoch']}")
            
            # Evaluate
            evaluator = RecognitionEvaluator(model, test_loader, categories, device)
            results = evaluator.full_evaluation(os.path.join(args.save_dir, 'results'))
        else:
            print(f"No checkpoint found at {checkpoint_path}")
    
    print("\n" + "="*60)
    print("COMPLETED")
    print("="*60)


if __name__ == '__main__':
    main()
