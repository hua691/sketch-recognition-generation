"""
Sketch Recognition 评估脚本
"""

import os
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json

from dataset import create_dataloaders
from model import SketchRecognitionModel


class Evaluator:
    def __init__(self, model, test_loader, categories, device='cpu'):
        self.model = model.to(device)
        self.model.eval()
        self.test_loader = test_loader
        self.categories = categories
        self.device = device
    
    @torch.no_grad()
    def evaluate(self):
        all_preds = []
        all_labels = []
        
        for images, labels in self.test_loader:
            images = images.to(self.device)
            outputs = self.model(images)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # 计算指标
        accuracy = (all_preds == all_labels).mean() * 100
        report = classification_report(all_labels, all_preds, 
                                       target_names=self.categories, 
                                       output_dict=True)
        cm = confusion_matrix(all_labels, all_preds)
        
        return {
            'accuracy': accuracy,
            'report': report,
            'confusion_matrix': cm.tolist(),
            'predictions': all_preds.tolist(),
            'labels': all_labels.tolist()
        }
    
    def plot_confusion_matrix(self, cm, save_path):
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.categories,
                   yticklabels=self.categories)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(save_path, dpi=100)
        plt.close()
    
    def full_evaluation(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        
        results = self.evaluate()
        
        # 保存结果
        with open(os.path.join(save_dir, 'recognition_results.json'), 'w') as f:
            json.dump({
                'accuracy': results['accuracy'],
                'report': results['report'],
                'confusion_matrix': results['confusion_matrix']
            }, f, indent=2)
        
        # 绘制混淆矩阵
        self.plot_confusion_matrix(
            np.array(results['confusion_matrix']),
            os.path.join(save_dir, 'confusion_matrix.png')
        )
        
        print("\n" + "="*50)
        print("RECOGNITION EVALUATION RESULTS")
        print("="*50)
        print(f"Test Accuracy: {results['accuracy']:.2f}%")
        print(f"\nPer-class metrics:")
        for cat in self.categories:
            if cat in results['report']:
                r = results['report'][cat]
                print(f"  {cat}: P={r['precision']:.3f}, R={r['recall']:.3f}, F1={r['f1-score']:.3f}")
        
        return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='D:/AAAAAAAAAAAAAproject/QuickDraw414k')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_recognition_model.pth')
    parser.add_argument('--save_dir', type=str, default='checkpoints/recognition_results')
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 加载checkpoint
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    config = checkpoint['config']
    categories = config['categories']
    
    # 加载数据
    _, _, test_loader, _ = create_dataloaders(
        args.data_dir,
        categories=categories,
        batch_size=args.batch_size,
        max_samples_per_class=config.get('max_samples', 500)
    )
    
    # 创建模型
    model = SketchRecognitionModel(
        num_classes=len(categories),
        backbone=config.get('backbone', 'resnet18')
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 评估
    evaluator = Evaluator(model, test_loader, categories, device)
    evaluator.full_evaluation(args.save_dir)


if __name__ == '__main__':
    main()
