"""
Sketch Generation 评估脚本
包含重建评估和生成质量评估
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

from dataset import create_dataloaders
from model import SketchVAE


class GenerationEvaluator:
    def __init__(self, model, test_loader, categories, device='cpu'):
        self.model = model.to(device)
        self.model.eval()
        self.test_loader = test_loader
        self.categories = categories
        self.device = device
    
    @torch.no_grad()
    def evaluate_reconstruction(self, num_samples=100):
        """评估重建质量"""
        print("Evaluating reconstruction...")
        
        total_mse = 0
        total_samples = 0
        
        for batch in tqdm(self.test_loader):
            sketch = batch['sketch'].to(self.device)
            seq_lens = batch['seq_len']
            
            # 重建
            reconstructed = self.model.reconstruct(sketch, seq_lens, temperature=0.1)
            
            # 计算MSE (只比较有效部分)
            for i in range(sketch.size(0)):
                seq_len = min(seq_lens[i].item(), reconstructed.size(1))
                orig = sketch[i, :seq_len, :2].cpu().numpy()
                recon = reconstructed[i, :seq_len, :2].cpu().numpy()
                
                # 转换为绝对坐标
                orig_abs = np.cumsum(orig, axis=0)
                recon_abs = np.cumsum(recon, axis=0)
                
                mse = np.mean((orig_abs - recon_abs) ** 2)
                total_mse += mse
                total_samples += 1
            
            if total_samples >= num_samples:
                break
        
        avg_mse = total_mse / total_samples
        print(f"Reconstruction MSE: {avg_mse:.6f}")
        
        return {'reconstruction_mse': avg_mse}
    
    @torch.no_grad()
    def evaluate_generation_diversity(self, num_samples=50):
        """评估生成多样性"""
        print("Evaluating generation diversity...")
        
        # 生成多个样本
        samples = self.model.generate(
            num_samples=num_samples,
            max_len=150,
            temperature=0.5,
            device=self.device
        )
        
        # 计算样本间的平均距离
        distances = []
        for i in range(num_samples):
            for j in range(i+1, num_samples):
                s1 = samples[i].cpu().numpy()
                s2 = samples[j].cpu().numpy()
                
                # 转换为绝对坐标
                abs1 = np.cumsum(s1[:, :2], axis=0)
                abs2 = np.cumsum(s2[:, :2], axis=0)
                
                # 计算DTW或简单的欧氏距离
                min_len = min(len(abs1), len(abs2))
                dist = np.mean(np.sqrt(np.sum((abs1[:min_len] - abs2[:min_len])**2, axis=1)))
                distances.append(dist)
        
        avg_diversity = np.mean(distances)
        std_diversity = np.std(distances)
        
        print(f"Generation Diversity: {avg_diversity:.4f} ± {std_diversity:.4f}")
        
        return {
            'diversity_mean': avg_diversity,
            'diversity_std': std_diversity
        }
    
    @torch.no_grad()
    def evaluate_interpolation(self, save_dir, num_steps=10):
        """评估潜在空间插值"""
        print("Evaluating latent interpolation...")
        
        # 获取两个样本
        batch = next(iter(self.test_loader))
        sketch1 = batch['sketch'][0:1].to(self.device)
        sketch2 = batch['sketch'][1:2].to(self.device)
        
        # 编码
        mu1, _ = self.model.encoder(sketch1)
        mu2, _ = self.model.encoder(sketch2)
        
        # 插值
        alphas = np.linspace(0, 1, num_steps)
        interpolated = []
        
        for alpha in alphas:
            z = (1 - alpha) * mu1 + alpha * mu2
            generated = self.model.decoder.generate(z, max_len=150, temperature=0.3, device=self.device)
            interpolated.append(generated[0].cpu().numpy())
        
        # 可视化
        fig, axes = plt.subplots(1, num_steps, figsize=(2*num_steps, 2))
        for i, (ax, sketch) in enumerate(zip(axes, interpolated)):
            self.draw_sketch(ax, sketch)
            ax.set_title(f'α={alphas[i]:.1f}')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'interpolation.png'), dpi=100)
        plt.close()
        
        print(f"Interpolation saved to {save_dir}/interpolation.png")
        
        return {'interpolation_steps': num_steps}
    
    @torch.no_grad()
    def generate_by_category(self, save_dir, samples_per_category=5):
        """按类别生成样本"""
        print("Generating samples by category...")
        
        os.makedirs(save_dir, exist_ok=True)
        
        # 收集每个类别的潜在向量
        category_latents = {cat: [] for cat in self.categories}
        
        for batch in self.test_loader:
            sketch = batch['sketch'].to(self.device)
            labels = batch['label'].numpy()
            
            mu, _ = self.model.encoder(sketch)
            
            for i, label in enumerate(labels):
                cat = self.categories[label]
                if len(category_latents[cat]) < 50:
                    category_latents[cat].append(mu[i].cpu())
        
        # 为每个类别生成样本
        for cat in self.categories:
            if len(category_latents[cat]) == 0:
                continue
            
            # 计算类别的平均潜在向量
            latents = torch.stack(category_latents[cat])
            mean_z = latents.mean(dim=0, keepdim=True).to(self.device)
            
            # 在均值附近采样
            noise = torch.randn(samples_per_category, mean_z.size(1), device=self.device) * 0.3
            z = mean_z + noise
            
            # 生成
            samples = self.model.decoder.generate(z, max_len=150, temperature=0.4, device=self.device)
            
            # 可视化
            fig, axes = plt.subplots(1, samples_per_category, figsize=(3*samples_per_category, 3))
            if samples_per_category == 1:
                axes = [axes]
            
            for i, ax in enumerate(axes):
                self.draw_sketch(ax, samples[i].cpu().numpy())
            
            plt.suptitle(f'Category: {cat}')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'generated_{cat}.png'), dpi=100)
            plt.close()
        
        print(f"Category samples saved to {save_dir}/")
    
    def draw_sketch(self, ax, sketch):
        """绘制草图"""
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
    
    def full_evaluation(self, save_dir):
        """完整评估"""
        os.makedirs(save_dir, exist_ok=True)
        
        results = {}
        
        # 重建评估
        recon_results = self.evaluate_reconstruction()
        results.update(recon_results)
        
        # 多样性评估
        diversity_results = self.evaluate_generation_diversity()
        results.update(diversity_results)
        
        # 插值评估
        interp_results = self.evaluate_interpolation(save_dir)
        results.update(interp_results)
        
        # 按类别生成
        self.generate_by_category(save_dir)
        
        # 保存结果
        with open(os.path.join(save_dir, 'generation_evaluation.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        for k, v in results.items():
            print(f"{k}: {v}")
        
        return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../sketch_generation_data')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_generation_model.pth')
    parser.add_argument('--save_dir', type=str, default='checkpoints/generation_results')
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 加载checkpoint
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    config = checkpoint['config']
    
    # 加载数据
    _, _, test_loader, categories = create_dataloaders(
        args.data_dir,
        categories=config.get('categories'),
        batch_size=args.batch_size,
        max_seq_len=config.get('max_seq_len', 200)
    )
    
    # 创建模型
    model = SketchVAE(
        input_dim=3,
        enc_hidden=config.get('enc_hidden', 256),
        dec_hidden=config.get('dec_hidden', 512),
        latent_dim=config.get('latent_dim', 128),
        num_mixtures=20
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 评估
    evaluator = GenerationEvaluator(model, test_loader, categories, device)
    evaluator.full_evaluation(args.save_dir)


if __name__ == '__main__':
    main()
