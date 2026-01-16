"""
Sketch Diffusion Models
Task 2B: SketchKnitter-style diffusion model for vectorized sketch generation
Reference: [8] SketchKnitter: Vectorized sketch generation with diffusion models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal position embeddings for diffusion timesteps."""
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class TransformerBlock(nn.Module):
    """Transformer block for sequence modeling."""
    
    def __init__(self, dim, heads=8, dim_head=64, mlp_dim=None, dropout=0.1):
        super().__init__()
        mlp_dim = mlp_dim or dim * 4
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x, mask=None):
        # Self-attention with residual
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, key_padding_mask=mask)
        x = x + attn_out
        
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        
        return x


class SketchDiffusionTransformer(nn.Module):
    """Transformer-based denoising network for sketch diffusion."""
    
    def __init__(self, input_dim=5, hidden_dim=256, num_layers=6, 
                 num_heads=8, max_len=200, num_classes=None, dropout=0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.max_len = max_len
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        # Class embedding (optional)
        self.num_classes = num_classes
        if num_classes is not None:
            self.class_embed = nn.Embedding(num_classes, hidden_dim)
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, hidden_dim) * 0.02)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, input_dim)
    
    def forward(self, x, t, class_label=None, mask=None):
        """
        Args:
            x: Noisy sketch sequence [batch, seq_len, input_dim]
            t: Diffusion timestep [batch]
            class_label: Optional class label [batch]
            mask: Optional padding mask [batch, seq_len]
        Returns:
            Predicted noise [batch, seq_len, input_dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # Project input
        h = self.input_proj(x)
        
        # Add positional embedding
        h = h + self.pos_embed[:, :seq_len]
        
        # Add time embedding
        t_emb = self.time_embed(t)
        h = h + t_emb.unsqueeze(1)
        
        # Add class embedding
        if class_label is not None and self.num_classes is not None:
            c_emb = self.class_embed(class_label)
            h = h + c_emb.unsqueeze(1)
        
        # Transformer layers
        for layer in self.layers:
            h = layer(h, mask)
        
        # Output
        h = self.output_norm(h)
        output = self.output_proj(h)
        
        return output


class SketchDiffusion(nn.Module):
    """DDPM-style diffusion model for sketch generation."""
    
    def __init__(self, denoiser, num_timesteps=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        
        self.denoiser = denoiser
        self.num_timesteps = num_timesteps
        
        # Define beta schedule
        betas = torch.linspace(beta_start, beta_end, num_timesteps)
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Register buffers
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1 - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas', torch.sqrt(1 / alphas))
        
        # For posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
    
    def q_sample(self, x_0, t, noise=None):
        """Forward diffusion process: q(x_t | x_0)."""
        if noise is None:
            noise = torch.randn_like(x_0)
        
        sqrt_alpha = self.sqrt_alphas_cumprod[t][:, None, None]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t][:, None, None]
        
        return sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise
    
    def p_losses(self, x_0, t, class_label=None, mask=None):
        """Compute training loss."""
        noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, t, noise)
        
        # Predict noise
        predicted_noise = self.denoiser(x_t, t, class_label, mask)
        
        # MSE loss
        loss = F.mse_loss(predicted_noise, noise, reduction='none')
        
        # Apply mask if provided
        if mask is not None:
            loss = loss * (~mask).unsqueeze(-1).float()
            loss = loss.sum() / (~mask).sum() / x_0.size(-1)
        else:
            loss = loss.mean()
        
        return loss
    
    @torch.no_grad()
    def p_sample(self, x_t, t, class_label=None):
        """Reverse diffusion step: p(x_{t-1} | x_t)."""
        # Predict noise
        predicted_noise = self.denoiser(x_t, t, class_label)
        
        # Get parameters
        beta_t = self.betas[t][:, None, None]
        sqrt_one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None]
        sqrt_recip_alpha_t = self.sqrt_recip_alphas[t][:, None, None]
        
        # Compute mean
        mean = sqrt_recip_alpha_t * (x_t - beta_t * predicted_noise / sqrt_one_minus_alpha_t)
        
        # Add noise (except for t=0)
        if t[0] > 0:
            noise = torch.randn_like(x_t)
            variance = self.posterior_variance[t][:, None, None]
            x_prev = mean + torch.sqrt(variance) * noise
        else:
            x_prev = mean
        
        return x_prev
    
    @torch.no_grad()
    def sample(self, shape, class_label=None, device='cuda'):
        """Generate samples using reverse diffusion."""
        batch_size = shape[0]
        
        # Start from pure noise
        x = torch.randn(shape).to(device)
        
        # Reverse diffusion
        for t in reversed(range(self.num_timesteps)):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            x = self.p_sample(x, t_batch, class_label)
        
        return x
    
    @torch.no_grad()
    def ddim_sample(self, shape, class_label=None, device='cuda', 
                    num_inference_steps=50, eta=0.0):
        """DDIM sampling for faster generation."""
        batch_size = shape[0]
        
        # Subset of timesteps
        step_size = self.num_timesteps // num_inference_steps
        timesteps = list(range(0, self.num_timesteps, step_size))[::-1]
        
        # Start from noise
        x = torch.randn(shape).to(device)
        
        for i, t in enumerate(timesteps):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            # Predict noise
            predicted_noise = self.denoiser(x, t_batch, class_label)
            
            # Get alpha values
            alpha_t = self.alphas_cumprod[t]
            alpha_prev = self.alphas_cumprod[timesteps[i+1]] if i < len(timesteps)-1 else torch.tensor(1.0)
            
            # Predict x_0
            x_0_pred = (x - torch.sqrt(1 - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)
            
            # Direction pointing to x_t
            sigma = eta * torch.sqrt((1 - alpha_prev) / (1 - alpha_t)) * torch.sqrt(1 - alpha_t / alpha_prev)
            
            # Compute x_{t-1}
            x = torch.sqrt(alpha_prev) * x_0_pred + \
                torch.sqrt(1 - alpha_prev - sigma**2) * predicted_noise
            
            if i < len(timesteps) - 1 and eta > 0:
                x = x + sigma * torch.randn_like(x)
        
        return x


class SketchKnitter(nn.Module):
    """SketchKnitter: Complete sketch generation pipeline with diffusion."""
    
    def __init__(self, num_classes, input_dim=5, hidden_dim=256, 
                 num_layers=6, max_len=200, num_timesteps=1000):
        super().__init__()
        
        self.num_classes = num_classes
        self.max_len = max_len
        self.input_dim = input_dim
        
        # Denoising network
        denoiser = SketchDiffusionTransformer(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes,
            max_len=max_len
        )
        
        # Diffusion model
        self.diffusion = SketchDiffusion(denoiser, num_timesteps)
    
    def forward(self, x, class_label=None, mask=None):
        """Training forward pass."""
        batch_size = x.size(0)
        device = x.device
        
        # Sample random timesteps
        t = torch.randint(0, self.diffusion.num_timesteps, (batch_size,), device=device)
        
        # Compute loss
        loss = self.diffusion.p_losses(x, t, class_label, mask)
        
        return loss
    
    @torch.no_grad()
    def generate(self, class_label, batch_size=1, seq_len=None, device='cuda', 
                 use_ddim=True, num_inference_steps=50):
        """Generate sketches for given class."""
        seq_len = seq_len or self.max_len
        shape = (batch_size, seq_len, self.input_dim)
        
        if use_ddim:
            samples = self.diffusion.ddim_sample(
                shape, class_label, device, num_inference_steps
            )
        else:
            samples = self.diffusion.sample(shape, class_label, device)
        
        return samples
    
    def reconstruct(self, x, class_label=None, noise_level=0.5):
        """Reconstruct sketch by adding noise and denoising."""
        device = x.device
        batch_size = x.size(0)
        
        # Add noise at specified level
        t = int(noise_level * self.diffusion.num_timesteps)
        t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
        
        x_noisy = self.diffusion.q_sample(x, t_batch)
        
        # Denoise
        for step in reversed(range(t)):
            t_step = torch.full((batch_size,), step, device=device, dtype=torch.long)
            x_noisy = self.diffusion.p_sample(x_noisy, t_step, class_label)
        
        return x_noisy


if __name__ == '__main__':
    # Test diffusion model
    batch_size = 4
    seq_len = 100
    num_classes = 10
    
    model = SketchKnitter(num_classes, max_len=seq_len)
    
    # Test training
    x = torch.randn(batch_size, seq_len, 5)
    labels = torch.randint(0, num_classes, (batch_size,))
    
    loss = model(x, labels)
    print(f"Training loss: {loss.item():.4f}")
    
    # Test generation
    model.eval()
    with torch.no_grad():
        generated = model.generate(labels[:1], batch_size=1, device='cpu', 
                                  use_ddim=True, num_inference_steps=10)
        print(f"Generated shape: {generated.shape}")
