"""
Controllable Sketch Generation Models
Task 2B: Train a multi-category sketch generation model producing sequential sketches
References: [2] Sketch-RNN (VAE), [7] Controllable stroke-based synthesis, [8] SketchKnitter (Diffusion)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class SketchRNNEncoder(nn.Module):
    """Bidirectional LSTM encoder for Sketch-RNN VAE."""
    
    def __init__(self, input_dim=5, hidden_dim=512, latent_dim=128, num_layers=1):
        super(SketchRNNEncoder, self).__init__()
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
                           batch_first=True, bidirectional=True)
        
        self.fc_mu = nn.Linear(hidden_dim * 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 2, latent_dim)
    
    def forward(self, x):
        # x: [batch, seq_len, 5]
        _, (h_n, _) = self.lstm(x)
        
        # Concatenate forward and backward hidden states
        h = torch.cat([h_n[-2], h_n[-1]], dim=1)
        
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        return mu, logvar


class SketchRNNDecoder(nn.Module):
    """Autoregressive LSTM decoder for Sketch-RNN."""
    
    def __init__(self, input_dim=5, hidden_dim=1024, latent_dim=128, 
                 num_layers=1, num_mixture=20):
        super(SketchRNNDecoder, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_mixture = num_mixture
        self.latent_dim = latent_dim
        
        # Initial state from latent
        self.fc_init_h = nn.Linear(latent_dim, hidden_dim * num_layers)
        self.fc_init_c = nn.Linear(latent_dim, hidden_dim * num_layers)
        
        # LSTM decoder
        self.lstm = nn.LSTM(input_dim + latent_dim, hidden_dim, num_layers, batch_first=True)
        
        # Output: GMM parameters for (dx, dy) + pen states
        # For each mixture: pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy
        # Plus 3 pen states: p1 (down), p2 (up), p3 (end)
        output_dim = num_mixture * 6 + 3
        self.fc_out = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, z, hidden=None):
        """
        Args:
            x: Input sequence [batch, seq_len, 5]
            z: Latent vector [batch, latent_dim]
            hidden: Optional initial hidden state
        """
        batch_size, seq_len, _ = x.shape
        
        # Initialize hidden state from z
        if hidden is None:
            h0 = self.fc_init_h(z).view(1, batch_size, self.hidden_dim)
            c0 = self.fc_init_c(z).view(1, batch_size, self.hidden_dim)
            hidden = (h0, c0)
        
        # Concatenate z to each input
        z_expanded = z.unsqueeze(1).expand(-1, seq_len, -1)
        lstm_input = torch.cat([x, z_expanded], dim=2)
        
        # LSTM forward
        lstm_out, hidden = self.lstm(lstm_input, hidden)
        
        # Output parameters
        output = self.fc_out(lstm_out)
        
        return output, hidden
    
    def get_mixture_params(self, output):
        """Extract GMM parameters from output."""
        M = self.num_mixture
        
        # Split output
        pi = output[:, :, :M]
        mu_x = output[:, :, M:2*M]
        mu_y = output[:, :, 2*M:3*M]
        sigma_x = output[:, :, 3*M:4*M]
        sigma_y = output[:, :, 4*M:5*M]
        rho_xy = output[:, :, 5*M:6*M]
        pen_logits = output[:, :, 6*M:]
        
        # Apply activations
        pi = F.softmax(pi, dim=-1)
        sigma_x = torch.exp(sigma_x)
        sigma_y = torch.exp(sigma_y)
        rho_xy = torch.tanh(rho_xy)
        pen_probs = F.softmax(pen_logits, dim=-1)
        
        return pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, pen_probs


class SketchRNNVAE(nn.Module):
    """Sketch-RNN VAE model for controllable sketch generation."""
    
    def __init__(self, input_dim=5, enc_hidden=512, dec_hidden=1024, 
                 latent_dim=128, num_mixture=20):
        super(SketchRNNVAE, self).__init__()
        
        self.encoder = SketchRNNEncoder(input_dim, enc_hidden, latent_dim)
        self.decoder = SketchRNNDecoder(input_dim, dec_hidden, latent_dim, 
                                        num_mixture=num_mixture)
        self.latent_dim = latent_dim
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        """
        Args:
            x: Input sketch sequence [batch, seq_len, 5]
        Returns:
            output: Decoder output parameters
            mu, logvar: Latent distribution parameters
        """
        # Encode
        mu, logvar = self.encoder(x)
        
        # Reparameterize
        z = self.reparameterize(mu, logvar)
        
        # Decode (teacher forcing)
        # Shift input for autoregressive decoding
        decoder_input = torch.zeros_like(x)
        decoder_input[:, 1:] = x[:, :-1]
        decoder_input[:, 0, 4] = 1  # Start token
        
        output, _ = self.decoder(decoder_input, z)
        
        return output, mu, logvar, z
    
    def generate(self, z=None, max_len=200, temperature=1.0, device='cuda'):
        """Generate sketch from latent vector."""
        if z is None:
            z = torch.randn(1, self.latent_dim).to(device)
        
        batch_size = z.size(0)
        
        # Start token
        stroke = torch.zeros(batch_size, 1, 5).to(device)
        stroke[:, 0, 4] = 1  # End state as start
        
        strokes = []
        hidden = None
        
        for _ in range(max_len):
            output, hidden = self.decoder(stroke, z, hidden)
            
            # Sample from GMM
            next_stroke = self._sample_next(output[:, -1], temperature)
            strokes.append(next_stroke)
            
            # Check for end
            if next_stroke[0, 4] > 0.5:
                break
            
            stroke = next_stroke.unsqueeze(1)
        
        return torch.cat(strokes, dim=0)
    
    def _sample_next(self, params, temperature=1.0):
        """Sample next stroke point from GMM."""
        pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, pen_probs = \
            self.decoder.get_mixture_params(params.unsqueeze(1))
        
        # Remove sequence dimension
        pi = pi.squeeze(1)
        mu_x = mu_x.squeeze(1)
        mu_y = mu_y.squeeze(1)
        sigma_x = sigma_x.squeeze(1) * temperature
        sigma_y = sigma_y.squeeze(1) * temperature
        rho_xy = rho_xy.squeeze(1)
        pen_probs = pen_probs.squeeze(1)
        
        # Sample mixture component
        pi_idx = torch.multinomial(pi, 1).squeeze(1)
        
        # Get parameters for selected component
        batch_idx = torch.arange(pi.size(0))
        mu_x_sel = mu_x[batch_idx, pi_idx]
        mu_y_sel = mu_y[batch_idx, pi_idx]
        sigma_x_sel = sigma_x[batch_idx, pi_idx]
        sigma_y_sel = sigma_y[batch_idx, pi_idx]
        rho_sel = rho_xy[batch_idx, pi_idx]
        
        # Sample from bivariate Gaussian
        dx, dy = self._sample_bivariate_gaussian(
            mu_x_sel, mu_y_sel, sigma_x_sel, sigma_y_sel, rho_sel
        )
        
        # Sample pen state
        pen_idx = torch.multinomial(pen_probs, 1).squeeze(1)
        pen_state = F.one_hot(pen_idx, 3).float()
        
        # Combine
        next_stroke = torch.zeros(pi.size(0), 5).to(pi.device)
        next_stroke[:, 0] = dx
        next_stroke[:, 1] = dy
        next_stroke[:, 2:] = pen_state
        
        return next_stroke
    
    def _sample_bivariate_gaussian(self, mu_x, mu_y, sigma_x, sigma_y, rho):
        """Sample from bivariate Gaussian distribution."""
        # Standard normal samples
        z1 = torch.randn_like(mu_x)
        z2 = torch.randn_like(mu_y)
        
        # Transform to correlated Gaussian
        x = mu_x + sigma_x * z1
        y = mu_y + sigma_y * (rho * z1 + torch.sqrt(1 - rho**2) * z2)
        
        return x, y
    
    def interpolate(self, x1, x2, num_steps=10):
        """Interpolate between two sketches in latent space."""
        mu1, _ = self.encoder(x1)
        mu2, _ = self.encoder(x2)
        
        interpolations = []
        for alpha in np.linspace(0, 1, num_steps):
            z = (1 - alpha) * mu1 + alpha * mu2
            sketch = self.generate(z)
            interpolations.append(sketch)
        
        return interpolations


class ConditionalSketchVAE(nn.Module):
    """Conditional VAE for class-conditioned sketch generation."""
    
    def __init__(self, num_classes, input_dim=5, enc_hidden=512, dec_hidden=1024,
                 latent_dim=128, num_mixture=20):
        super(ConditionalSketchVAE, self).__init__()
        
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        
        # Class embedding
        self.class_embed = nn.Embedding(num_classes, 64)
        
        # Encoder with class conditioning
        self.encoder = SketchRNNEncoder(input_dim, enc_hidden, latent_dim)
        self.fc_mu = nn.Linear(latent_dim + 64, latent_dim)
        self.fc_logvar = nn.Linear(latent_dim + 64, latent_dim)
        
        # Decoder with class conditioning
        self.decoder = SketchRNNDecoder(input_dim, dec_hidden, latent_dim + 64,
                                        num_mixture=num_mixture)
    
    def forward(self, x, labels):
        """Forward pass with class labels."""
        # Get class embedding
        class_emb = self.class_embed(labels)
        
        # Encode
        mu_enc, logvar_enc = self.encoder(x)
        
        # Condition on class
        mu_cond = torch.cat([mu_enc, class_emb], dim=1)
        mu = self.fc_mu(mu_cond)
        logvar = self.fc_logvar(mu_cond)
        
        # Reparameterize
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        # Condition z on class for decoder
        z_cond = torch.cat([z, class_emb], dim=1)
        
        # Decode
        decoder_input = torch.zeros_like(x)
        decoder_input[:, 1:] = x[:, :-1]
        decoder_input[:, 0, 4] = 1
        
        output, _ = self.decoder(decoder_input, z_cond)
        
        return output, mu, logvar
    
    def generate(self, label, z=None, max_len=200, temperature=1.0, device='cuda'):
        """Generate sketch for given class."""
        if z is None:
            z = torch.randn(1, self.latent_dim).to(device)
        
        class_emb = self.class_embed(label.to(device))
        z_cond = torch.cat([z, class_emb], dim=1)
        
        # Generate using decoder
        stroke = torch.zeros(1, 1, 5).to(device)
        stroke[:, 0, 4] = 1
        
        strokes = []
        hidden = None
        
        for _ in range(max_len):
            output, hidden = self.decoder(stroke, z_cond, hidden)
            next_stroke = self._sample_next(output[:, -1], temperature)
            strokes.append(next_stroke)
            
            if next_stroke[0, 4] > 0.5:
                break
            
            stroke = next_stroke.unsqueeze(1)
        
        return torch.cat(strokes, dim=0) if strokes else stroke.squeeze(1)


def sketch_rnn_loss(output, target, mu, logvar, decoder, kl_weight=0.5):
    """Compute Sketch-RNN loss: reconstruction + KL divergence."""
    
    # Get GMM parameters
    pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, pen_probs = \
        decoder.get_mixture_params(output)
    
    # Target values
    dx = target[:, :, 0:1]
    dy = target[:, :, 1:2]
    pen = target[:, :, 2:]
    
    # Reconstruction loss for (dx, dy) using GMM
    # Bivariate Gaussian log probability
    z_x = (dx - mu_x) / sigma_x
    z_y = (dy - mu_y) / sigma_y
    
    z = z_x**2 + z_y**2 - 2 * rho_xy * z_x * z_y
    denom = 2 * (1 - rho_xy**2)
    
    log_n = -z / denom - torch.log(2 * np.pi * sigma_x * sigma_y * torch.sqrt(1 - rho_xy**2))
    log_prob = torch.logsumexp(torch.log(pi + 1e-10) + log_n, dim=-1)
    
    recon_loss_xy = -log_prob.mean()
    
    # Reconstruction loss for pen state
    recon_loss_pen = F.cross_entropy(
        pen_probs.reshape(-1, 3), 
        pen.argmax(dim=-1).reshape(-1)
    )
    
    # KL divergence
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Total loss
    total_loss = recon_loss_xy + recon_loss_pen + kl_weight * kl_loss
    
    return total_loss, recon_loss_xy, recon_loss_pen, kl_loss


if __name__ == '__main__':
    # Test models
    batch_size = 4
    seq_len = 100
    
    # Test Sketch-RNN VAE
    model = SketchRNNVAE()
    x = torch.randn(batch_size, seq_len, 5)
    output, mu, logvar, z = model(x)
    print(f"SketchRNN VAE output: {output.shape}, z: {z.shape}")
    
    # Test generation
    model.eval()
    with torch.no_grad():
        generated = model.generate(max_len=50, device='cpu')
        print(f"Generated sketch: {generated.shape}")
    
    # Test Conditional VAE
    num_classes = 10
    cond_model = ConditionalSketchVAE(num_classes)
    labels = torch.randint(0, num_classes, (batch_size,))
    output, mu, logvar = cond_model(x, labels)
    print(f"Conditional VAE output: {output.shape}")
