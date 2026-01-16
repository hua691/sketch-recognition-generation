"""
Evaluation script for Sketch Recognition and Generation
Includes metrics for both tasks
"""

import os
import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
from scipy import stats
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from data.quickdraw_dataset import QuickDrawDataset, create_dataloaders
from models.sketch_recognition import get_recognition_model
from models.sketch_generation import SketchRNNVAE, ConditionalSketchVAE


class RecognitionEvaluator:
    """Evaluator for sketch recognition models."""
    
    def __init__(self, model, test_loader, categories, device='cuda'):
        self.model = model.to(device)
        self.test_loader = test_loader
        self.categories = categories
        self.device = device
        self.model.eval()
    
    @torch.no_grad()
    def predict(self):
        all_preds, all_labels, all_probs = [], [], []
        
        for batch in tqdm(self.test_loader, desc='Evaluating'):
            if len(batch) == 2:
                data, labels = batch
                data = data.to(self.device)
            else:
                seq, img, labels = batch
                seq, img = seq.to(self.device), img.to(self.device)
                data = (seq, img)
            
            if isinstance(data, tuple):
                outputs = self.model(*data)
            else:
                outputs = self.model(data)
            
            probs = F.softmax(outputs, dim=1)
            _, preds = outputs.max(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
        
        return np.array(all_preds), np.array(all_labels), np.array(all_probs)
    
    def compute_metrics(self, y_true, y_pred):
        return {
            'accuracy': accuracy_score(y_true, y_pred) * 100,
            'precision': precision_score(y_true, y_pred, average='weighted') * 100,
            'recall': recall_score(y_true, y_pred, average='weighted') * 100,
            'f1_score': f1_score(y_true, y_pred, average='weighted') * 100,
        }
    
    def bootstrap_ci(self, y_true, y_pred, n_bootstrap=1000, confidence=0.95):
        """Compute confidence intervals using bootstrap."""
        n = len(y_true)
        accuracies = []
        
        np.random.seed(42)
        for _ in range(n_bootstrap):
            idx = np.random.choice(n, n, replace=True)
            acc = accuracy_score(y_true[idx], y_pred[idx])
            accuracies.append(acc)
        
        accuracies = np.array(accuracies) * 100
        alpha = 1 - confidence
        
        return {
            'mean': np.mean(accuracies),
            'std': np.std(accuracies),
            'ci_lower': np.percentile(accuracies, alpha/2 * 100),
            'ci_upper': np.percentile(accuracies, (1-alpha/2) * 100)
        }
    
    def plot_confusion_matrix(self, y_true, y_pred, save_path):
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.categories, yticklabels=self.categories)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
    
    def full_evaluation(self, save_dir='results'):
        os.makedirs(save_dir, exist_ok=True)
        
        y_pred, y_true, y_prob = self.predict()
        
        metrics = self.compute_metrics(y_true, y_pred)
        ci = self.bootstrap_ci(y_true, y_pred)
        
        self.plot_confusion_matrix(y_true, y_pred, 
                                  os.path.join(save_dir, 'confusion_matrix.png'))
        
        results = {
            'metrics': metrics,
            'confidence_interval': ci,
            'n_samples': len(y_true),
            'n_classes': len(self.categories)
        }
        
        with open(os.path.join(save_dir, 'recognition_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        print("\n" + "="*50)
        print("RECOGNITION EVALUATION RESULTS")
        print("="*50)
        print(f"Test Samples: {len(y_true)}")
        print(f"Classes: {len(self.categories)}")
        print(f"\nMetrics:")
        for name, value in metrics.items():
            print(f"  {name}: {value:.2f}%")
        print(f"\n95% Confidence Interval:")
        print(f"  Accuracy: {ci['mean']:.2f}% ± {ci['std']:.2f}%")
        print(f"  CI: [{ci['ci_lower']:.2f}%, {ci['ci_upper']:.2f}%]")
        
        return results


class GenerationEvaluator:
    """Evaluator for sketch generation models."""
    
    def __init__(self, gen_model, recog_model, categories, device='cuda'):
        self.gen_model = gen_model.to(device)
        self.recog_model = recog_model.to(device) if recog_model else None
        self.categories = categories
        self.device = device
        self.gen_model.eval()
        if self.recog_model:
            self.recog_model.eval()
    
    @torch.no_grad()
    def evaluate_reconstruction(self, test_loader, n_samples=100):
        """Evaluate reconstruction quality."""
        total_mse = 0
        n = 0
        
        for batch in test_loader:
            if len(batch) == 2:
                data, labels = batch
            else:
                data, _, labels = batch
            
            data = data.to(self.device)
            labels = labels.to(self.device)
            
            # Encode and decode
            if hasattr(self.gen_model, 'class_embed'):
                output, mu, logvar = self.gen_model(data, labels)
            else:
                output, mu, logvar, z = self.gen_model(data)
            
            # Get reconstructed coordinates
            pi, mu_x, mu_y, _, _, _, _ = self.gen_model.decoder.get_mixture_params(output)
            
            # Use mean of most likely mixture
            idx = pi.argmax(dim=-1, keepdim=True)
            recon_x = torch.gather(mu_x, -1, idx).squeeze(-1)
            recon_y = torch.gather(mu_y, -1, idx).squeeze(-1)
            
            # MSE
            mse = ((recon_x - data[:, :, 0])**2 + (recon_y - data[:, :, 1])**2).mean()
            total_mse += mse.item() * data.size(0)
            n += data.size(0)
            
            if n >= n_samples:
                break
        
        return total_mse / n
    
    @torch.no_grad()
    def evaluate_category_accuracy(self, n_samples_per_class=50):
        """Evaluate if generated sketches are recognized as correct category."""
        if self.recog_model is None:
            return None
        
        correct = 0
        total = 0
        
        for class_idx in range(len(self.categories)):
            label = torch.tensor([class_idx]).to(self.device)
            
            for _ in range(n_samples_per_class):
                # Generate
                if hasattr(self.gen_model, 'class_embed'):
                    z = torch.randn(1, self.gen_model.latent_dim).to(self.device)
                    sketch = self.gen_model.generate(label, z, max_len=100, device=self.device)
                else:
                    z = torch.randn(1, self.gen_model.latent_dim).to(self.device)
                    sketch = self.gen_model.generate(z, max_len=100, device=self.device)
                
                # Recognize
                sketch_input = sketch.unsqueeze(0)
                if sketch_input.size(1) < 200:
                    pad = torch.zeros(1, 200 - sketch_input.size(1), 5).to(self.device)
                    sketch_input = torch.cat([sketch_input, pad], dim=1)
                
                pred = self.recog_model(sketch_input[:, :, :3]).argmax(dim=1)
                
                if pred.item() == class_idx:
                    correct += 1
                total += 1
        
        return correct / total * 100
    
    @torch.no_grad()
    def evaluate_diversity(self, n_samples=100):
        """Evaluate diversity of generated sketches using latent space."""
        latents = []
        
        for _ in range(n_samples):
            z = torch.randn(1, self.gen_model.latent_dim).to(self.device)
            latents.append(z.cpu().numpy())
        
        latents = np.concatenate(latents, axis=0)
        
        # Compute pairwise distances
        from scipy.spatial.distance import pdist
        distances = pdist(latents)
        
        return {
            'mean_distance': np.mean(distances),
            'std_distance': np.std(distances),
            'min_distance': np.min(distances),
            'max_distance': np.max(distances)
        }
    
    def compute_fid_like_metric(self, real_loader, n_samples=500):
        """Compute FID-like metric using recognition model features."""
        if self.recog_model is None:
            return None
        
        # Get features from real sketches
        real_features = []
        n = 0
        for batch in real_loader:
            if len(batch) == 2:
                data, _ = batch
            else:
                data, _, _ = batch
            
            data = data.to(self.device)
            
            # Get features before classifier
            with torch.no_grad():
                if hasattr(self.recog_model, 'lstm'):
                    out, _ = self.recog_model.lstm(data)
                    feat = out[:, -1]
                else:
                    feat = self.recog_model.backbone(data)
                    if hasattr(feat, 'flatten'):
                        feat = feat.flatten(1)
            
            real_features.append(feat.cpu().numpy())
            n += data.size(0)
            if n >= n_samples:
                break
        
        real_features = np.concatenate(real_features, axis=0)[:n_samples]
        
        # Get features from generated sketches
        gen_features = []
        for i in range(n_samples):
            z = torch.randn(1, self.gen_model.latent_dim).to(self.device)
            
            if hasattr(self.gen_model, 'class_embed'):
                label = torch.tensor([i % len(self.categories)]).to(self.device)
                sketch = self.gen_model.generate(label, z, max_len=100, device=self.device)
            else:
                sketch = self.gen_model.generate(z, max_len=100, device=self.device)
            
            sketch_input = sketch.unsqueeze(0)
            if sketch_input.size(1) < 200:
                pad = torch.zeros(1, 200 - sketch_input.size(1), 5).to(self.device)
                sketch_input = torch.cat([sketch_input, pad], dim=1)
            
            with torch.no_grad():
                if hasattr(self.recog_model, 'lstm'):
                    out, _ = self.recog_model.lstm(sketch_input[:, :, :3])
                    feat = out[:, -1]
                else:
                    feat = self.recog_model.backbone(sketch_input)
            
            gen_features.append(feat.cpu().numpy())
        
        gen_features = np.concatenate(gen_features, axis=0)
        
        # Compute FID-like metric
        mu_real, sigma_real = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
        mu_gen, sigma_gen = np.mean(gen_features, axis=0), np.cov(gen_features, rowvar=False)
        
        diff = mu_real - mu_gen
        
        # Simplified FID (without matrix sqrt)
        fid = np.sum(diff**2) + np.trace(sigma_real + sigma_gen - 2*np.sqrt(sigma_real @ sigma_gen + 1e-6))
        
        return fid
    
    def full_evaluation(self, test_loader, save_dir='results'):
        os.makedirs(save_dir, exist_ok=True)
        
        results = {}
        
        # Reconstruction
        print("Evaluating reconstruction...")
        recon_mse = self.evaluate_reconstruction(test_loader)
        results['reconstruction_mse'] = recon_mse
        
        # Category accuracy
        if self.recog_model:
            print("Evaluating category accuracy...")
            cat_acc = self.evaluate_category_accuracy(n_samples_per_class=20)
            results['category_accuracy'] = cat_acc
        
        # Diversity
        print("Evaluating diversity...")
        diversity = self.evaluate_diversity()
        results['diversity'] = diversity
        
        with open(os.path.join(save_dir, 'generation_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        print("\n" + "="*50)
        print("GENERATION EVALUATION RESULTS")
        print("="*50)
        print(f"Reconstruction MSE: {recon_mse:.6f}")
        if 'category_accuracy' in results:
            print(f"Category Accuracy: {results['category_accuracy']:.2f}%")
        print(f"Diversity (mean distance): {diversity['mean_distance']:.4f}")
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate Sketch Models')
    parser.add_argument('--task', type=str, default='recognition',
                       choices=['recognition', 'generation', 'both'])
    parser.add_argument('--data_dir', type=str, default='./quickdraw_data')
    parser.add_argument('--categories', type=str, nargs='+',
                       default=['cat', 'dog', 'car', 'house', 'tree'])
    parser.add_argument('--recog_checkpoint', type=str, 
                       default='checkpoints/best_recognition_model.pth')
    parser.add_argument('--gen_checkpoint', type=str,
                       default='checkpoints/best_vae_model.pth')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--save_dir', type=str, default='results')
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if args.task in ['recognition', 'both']:
        # Load recognition model
        checkpoint = torch.load(args.recog_checkpoint, map_location='cpu')
        config = checkpoint.get('config', {})
        
        num_classes = len(args.categories)
        model = get_recognition_model(
            config.get('model', 'transformer'), 
            num_classes
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Create test loader
        _, _, test_loader, categories = create_dataloaders(
            args.data_dir, args.categories, args.batch_size,
            config.get('representation', 'stroke3')
        )
        
        evaluator = RecognitionEvaluator(model, test_loader, categories, device)
        evaluator.full_evaluation(args.save_dir)
    
    if args.task in ['generation', 'both']:
        # Load generation model
        checkpoint = torch.load(args.gen_checkpoint, map_location='cpu')
        config = checkpoint.get('config', {})
        
        num_classes = len(args.categories)
        gen_model = ConditionalSketchVAE(num_classes, latent_dim=config.get('latent_dim', 128))
        gen_model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load recognition model for evaluation
        recog_model = None
        if os.path.exists(args.recog_checkpoint):
            recog_ckpt = torch.load(args.recog_checkpoint, map_location='cpu')
            recog_model = get_recognition_model('rnn', num_classes)
            recog_model.load_state_dict(recog_ckpt['model_state_dict'])
        
        # Create test loader
        _, _, test_loader, categories = create_dataloaders(
            args.data_dir, args.categories, args.batch_size, 'stroke5'
        )
        
        evaluator = GenerationEvaluator(gen_model, recog_model, categories, device)
        evaluator.full_evaluation(test_loader, args.save_dir)


if __name__ == '__main__':
    main()
