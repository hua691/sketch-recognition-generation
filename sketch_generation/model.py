"""
Sketch-RNN VAE Model for Sketch Generation
基于 "A Neural Representation of Sketch Drawings" (Ha & Eck, 2017)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class SketchEncoder(nn.Module):
    """双向LSTM编码器"""
    
    def __init__(self, input_dim=3, hidden_dim=256, latent_dim=128, num_layers=1, dropout=0.1):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 从双向LSTM输出映射到潜在空间
        self.fc_mu = nn.Linear(hidden_dim * 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 2, latent_dim)
    
    def forward(self, x, seq_lens=None):
        """
        Args:
            x: [batch, seq_len, 3] 草图序列
            seq_lens: [batch] 每个序列的实际长度
        Returns:
            mu, logvar: [batch, latent_dim]
        """
        batch_size = x.size(0)
        
        # LSTM编码
        if seq_lens is not None:
            # Pack序列
            packed = nn.utils.rnn.pack_padded_sequence(
                x, seq_lens.cpu(), batch_first=True, enforce_sorted=False
            )
            _, (h_n, _) = self.lstm(packed)
        else:
            _, (h_n, _) = self.lstm(x)
        
        # 合并双向隐藏状态 [num_layers*2, batch, hidden] -> [batch, hidden*2]
        h_n = h_n.view(self.num_layers, 2, batch_size, self.hidden_dim)
        h_n = h_n[-1]  # 取最后一层
        h_n = torch.cat([h_n[0], h_n[1]], dim=-1)  # [batch, hidden*2]
        
        # 映射到潜在空间
        mu = self.fc_mu(h_n)
        logvar = self.fc_logvar(h_n)
        
        return mu, logvar


class SketchDecoder(nn.Module):
    """自回归LSTM解码器，输出GMM参数"""
    
    def __init__(self, input_dim=3, hidden_dim=512, latent_dim=128, 
                 num_layers=1, num_mixtures=20, dropout=0.1):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.num_mixtures = num_mixtures
        
        # 从潜在向量初始化隐藏状态
        self.fc_init_h = nn.Linear(latent_dim, hidden_dim * num_layers)
        self.fc_init_c = nn.Linear(latent_dim, hidden_dim * num_layers)
        
        # 解码器LSTM
        self.lstm = nn.LSTM(
            input_dim + latent_dim,  # 输入 = 上一步输出 + 潜在向量
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # GMM输出层: 每个混合成分有 (pi, mu_x, mu_y, sigma_x, sigma_y, rho)
        # 加上3个pen状态 (p1, p2, p3)
        output_dim = num_mixtures * 6 + 3
        self.fc_out = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, z, hidden=None):
        """
        Args:
            x: [batch, seq_len, 3] 输入序列 (teacher forcing)
            z: [batch, latent_dim] 潜在向量
        Returns:
            outputs: GMM参数
        """
        batch_size, seq_len, _ = x.size()
        
        # 初始化隐藏状态
        if hidden is None:
            h_0 = self.fc_init_h(z).view(batch_size, self.num_layers, self.hidden_dim)
            h_0 = h_0.permute(1, 0, 2).contiguous()
            c_0 = self.fc_init_c(z).view(batch_size, self.num_layers, self.hidden_dim)
            c_0 = c_0.permute(1, 0, 2).contiguous()
            hidden = (h_0, c_0)
        
        # 将z拼接到每个时间步的输入
        z_expanded = z.unsqueeze(1).expand(-1, seq_len, -1)
        lstm_input = torch.cat([x, z_expanded], dim=-1)
        
        # LSTM解码
        lstm_out, hidden = self.lstm(lstm_input, hidden)
        
        # 输出GMM参数
        outputs = self.fc_out(lstm_out)
        
        return outputs, hidden
    
    def generate(self, z, max_len=200, temperature=1.0, device='cpu'):
        """
        自回归生成草图
        
        Args:
            z: [batch, latent_dim] 潜在向量
            max_len: 最大生成长度
            temperature: 采样温度
        Returns:
            sketches: [batch, seq_len, 3]
        """
        batch_size = z.size(0)
        
        # 初始化
        h = self.fc_init_h(z).view(batch_size, self.num_layers, self.hidden_dim)
        h = h.permute(1, 0, 2).contiguous()
        c = self.fc_init_c(z).view(batch_size, self.num_layers, self.hidden_dim)
        c = c.permute(1, 0, 2).contiguous()
        hidden = (h, c)
        
        # 起始token [0, 0, 1, 0, 0] -> [0, 0, 0] (pen down)
        current_input = torch.zeros(batch_size, 1, 3, device=device)
        
        sketches = []
        
        for _ in range(max_len):
            # 拼接z
            z_step = z.unsqueeze(1)
            lstm_input = torch.cat([current_input, z_step], dim=-1)
            
            # LSTM步进
            lstm_out, hidden = self.lstm(lstm_input, hidden)
            output = self.fc_out(lstm_out)
            
            # 采样下一步
            next_point = self.sample_next(output.squeeze(1), temperature)
            sketches.append(next_point)
            
            # 检查是否结束 (pen_state == 1 表示结束)
            # 更新输入
            current_input = next_point.unsqueeze(1)
        
        return torch.stack(sketches, dim=1)
    
    def sample_next(self, output, temperature=1.0):
        """从GMM中采样下一个点"""
        M = self.num_mixtures
        
        # 解析输出
        pi = output[:, :M]
        mu_x = output[:, M:2*M]
        mu_y = output[:, 2*M:3*M]
        sigma_x = torch.exp(output[:, 3*M:4*M])
        sigma_y = torch.exp(output[:, 4*M:5*M])
        rho = torch.tanh(output[:, 5*M:6*M])
        pen_logits = output[:, 6*M:]
        
        # 应用温度
        pi = F.softmax(pi / temperature, dim=-1)
        
        # 选择混合成分
        idx = torch.multinomial(pi, 1).squeeze(-1)
        batch_idx = torch.arange(output.size(0), device=output.device)
        
        # 获取选中成分的参数
        mu_x_sel = mu_x[batch_idx, idx]
        mu_y_sel = mu_y[batch_idx, idx]
        sigma_x_sel = sigma_x[batch_idx, idx] * temperature
        sigma_y_sel = sigma_y[batch_idx, idx] * temperature
        rho_sel = rho[batch_idx, idx]
        
        # 从二元高斯采样
        mean = torch.stack([mu_x_sel, mu_y_sel], dim=-1)
        
        # 构建协方差矩阵
        cov_xx = sigma_x_sel ** 2
        cov_yy = sigma_y_sel ** 2
        cov_xy = rho_sel * sigma_x_sel * sigma_y_sel
        
        # 采样
        eps = torch.randn(2, device=output.device)
        dx = mu_x_sel + sigma_x_sel * eps[0]
        dy = mu_y_sel + sigma_y_sel * (rho_sel * eps[0] + torch.sqrt(1 - rho_sel**2 + 1e-6) * eps[1])
        
        # 采样pen状态
        pen_probs = F.softmax(pen_logits / temperature, dim=-1)
        pen_idx = torch.multinomial(pen_probs, 1).squeeze(-1)
        pen_state = (pen_idx > 0).float()  # 简化为0/1
        
        return torch.stack([dx, dy, pen_state], dim=-1)


class SketchVAE(nn.Module):
    """Sketch-RNN VAE完整模型"""
    
    def __init__(self, input_dim=3, enc_hidden=256, dec_hidden=512,
                 latent_dim=128, num_layers=1, num_mixtures=20, dropout=0.1):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.num_mixtures = num_mixtures
        
        self.encoder = SketchEncoder(input_dim, enc_hidden, latent_dim, num_layers, dropout)
        self.decoder = SketchDecoder(input_dim, dec_hidden, latent_dim, num_layers, num_mixtures, dropout)
    
    def reparameterize(self, mu, logvar):
        """重参数化技巧"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x, seq_lens=None):
        """
        Args:
            x: [batch, seq_len, 3]
        Returns:
            outputs, mu, logvar
        """
        # 编码
        mu, logvar = self.encoder(x, seq_lens)
        
        # 重参数化
        z = self.reparameterize(mu, logvar)
        
        # 解码 (teacher forcing: 输入是x[:-1], 目标是x[1:])
        decoder_input = x[:, :-1, :]
        outputs, _ = self.decoder(decoder_input, z)
        
        return outputs, mu, logvar
    
    def generate(self, num_samples=1, max_len=200, temperature=1.0, 
                 z=None, device='cpu'):
        """生成草图"""
        if z is None:
            z = torch.randn(num_samples, self.latent_dim, device=device)
        
        return self.decoder.generate(z, max_len, temperature, device)
    
    def reconstruct(self, x, seq_lens=None, temperature=0.1):
        """重建草图"""
        mu, logvar = self.encoder(x, seq_lens)
        z = mu  # 使用均值进行重建
        return self.decoder.generate(z, x.size(1), temperature, x.device)


def compute_loss(outputs, targets, mu, logvar, mask=None, 
                 kl_weight=0.5, num_mixtures=20):
    """
    计算VAE损失 = 重建损失 + KL散度
    
    Args:
        outputs: [batch, seq_len, output_dim] 解码器输出
        targets: [batch, seq_len, 3] 目标序列
        mu, logvar: 潜在分布参数
        mask: [batch, seq_len] 有效位置mask
        kl_weight: KL散度权重
    """
    batch_size, seq_len, _ = targets.size()
    M = num_mixtures
    
    # 解析GMM参数
    pi = outputs[:, :, :M]
    mu_x = outputs[:, :, M:2*M]
    mu_y = outputs[:, :, 2*M:3*M]
    sigma_x = torch.exp(outputs[:, :, 3*M:4*M])
    sigma_y = torch.exp(outputs[:, :, 4*M:5*M])
    rho = torch.tanh(outputs[:, :, 5*M:6*M])
    pen_logits = outputs[:, :, 6*M:]
    
    # 目标值
    dx = targets[:, :, 0:1]
    dy = targets[:, :, 1:2]
    pen = targets[:, :, 2].long()
    
    # 计算二元高斯对数概率
    z_x = (dx - mu_x) / (sigma_x + 1e-6)
    z_y = (dy - mu_y) / (sigma_y + 1e-6)
    
    z_xy = z_x**2 + z_y**2 - 2*rho*z_x*z_y
    z_xy = z_xy / (1 - rho**2 + 1e-6)
    
    log_norm = -z_xy / 2 - torch.log(2 * 3.14159 * sigma_x * sigma_y * torch.sqrt(1 - rho**2 + 1e-6) + 1e-6)
    
    # 混合权重
    log_pi = F.log_softmax(pi, dim=-1)
    
    # Log-sum-exp
    log_prob = torch.logsumexp(log_pi + log_norm, dim=-1)
    
    # Pen状态损失
    pen_loss = F.cross_entropy(pen_logits.view(-1, 3), pen.view(-1), reduction='none')
    pen_loss = pen_loss.view(batch_size, seq_len)
    
    # 总重建损失
    recon_loss = -log_prob + pen_loss
    
    if mask is not None:
        recon_loss = (recon_loss * mask).sum() / (mask.sum() + 1e-6)
    else:
        recon_loss = recon_loss.mean()
    
    # KL散度
    kl_loss = -0.5 * torch.mean(1 + logvar - mu**2 - torch.exp(logvar))
    
    # 总损失
    total_loss = recon_loss + kl_weight * kl_loss
    
    return total_loss, recon_loss, kl_loss


if __name__ == '__main__':
    # 测试模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = SketchVAE(
        input_dim=3,
        enc_hidden=256,
        dec_hidden=512,
        latent_dim=128,
        num_mixtures=20
    ).to(device)
    
    # 测试前向传播
    x = torch.randn(4, 100, 3).to(device)
    outputs, mu, logvar = model(x)
    print(f"Output shape: {outputs.shape}")
    print(f"Mu shape: {mu.shape}")
    
    # 测试生成
    generated = model.generate(num_samples=2, max_len=50, device=device)
    print(f"Generated shape: {generated.shape}")
