"""
Sketch Recognition Models
Task 2A: Develop a classification model for QuickDraw-414k dataset
References: [1] Multigraph Transformer, [3] Sketch-a-Net, [5] Sketch-R2CNN, [6] SketchMLP
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import math


class SketchCNN(nn.Module):
    """CNN-based sketch recognition for rasterized images."""
    
    def __init__(self, num_classes, pretrained=True, backbone='resnet50'):
        super(SketchCNN, self).__init__()
        
        if backbone == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            self.backbone.conv1 = nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)
            num_features = 512
        elif backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            self.backbone.conv1 = nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)
            num_features = 2048
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)


class SketchRNN(nn.Module):
    """RNN-based sketch recognition for stroke sequences (Sketch-R2CNN style)."""
    
    def __init__(self, num_classes, input_dim=3, hidden_dim=512, num_layers=2, 
                 bidirectional=True, dropout=0.3):
        super(SketchRNN, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, bidirectional=bidirectional, dropout=dropout
        )
        
        fc_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        self.attention = nn.Sequential(
            nn.Linear(fc_input_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(fc_input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        lstm_out, _ = self.lstm(x)  # [batch, seq_len, hidden*2]
        
        # Attention mechanism
        attn_weights = self.attention(lstm_out)  # [batch, seq_len, 1]
        attn_weights = F.softmax(attn_weights, dim=1)
        
        # Weighted sum
        context = torch.sum(attn_weights * lstm_out, dim=1)  # [batch, hidden*2]
        
        return self.classifier(context)


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer."""
    
    def __init__(self, d_model, max_len=500, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class SketchTransformer(nn.Module):
    """Transformer-based sketch recognition (inspired by Multigraph Transformer)."""
    
    def __init__(self, num_classes, input_dim=3, d_model=256, nhead=8, 
                 num_layers=6, dim_feedforward=1024, dropout=0.1, max_len=200):
        super(SketchTransformer, self).__init__()
        
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
    
    def forward(self, x, mask=None):
        # x: [batch, seq_len, input_dim]
        batch_size = x.size(0)
        
        # Project input
        x = self.input_proj(x)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoding
        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        
        # Use CLS token for classification
        cls_output = x[:, 0]
        
        return self.classifier(cls_output)


class SketchMLP(nn.Module):
    """MLP-based sketch recognition combining raster and sequence features."""
    
    def __init__(self, num_classes, seq_len=200, input_dim=3, img_size=224, hidden_dim=512):
        super(SketchMLP, self).__init__()
        
        # Sequence branch (MLP-Mixer style)
        self.seq_proj = nn.Linear(input_dim, hidden_dim)
        self.seq_mixer = nn.Sequential(
            nn.Linear(seq_len, seq_len),
            nn.GELU(),
            nn.Linear(seq_len, seq_len)
        )
        self.channel_mixer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        self.seq_norm = nn.LayerNorm(hidden_dim)
        
        # Image branch
        self.img_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # Fusion and classification
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + 128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, seq, img):
        # Sequence branch
        seq_feat = self.seq_proj(seq)  # [B, L, H]
        seq_feat = seq_feat + self.seq_mixer(seq_feat.transpose(1, 2)).transpose(1, 2)
        seq_feat = self.seq_norm(seq_feat + self.channel_mixer(seq_feat))
        seq_feat = seq_feat.mean(dim=1)  # [B, H]
        
        # Image branch
        img_feat = self.img_encoder(img).flatten(1)  # [B, 128]
        
        # Fusion
        combined = torch.cat([seq_feat, img_feat], dim=1)
        
        return self.classifier(combined)


class HybridSketchNet(nn.Module):
    """Hybrid model combining CNN and RNN for sketch recognition."""
    
    def __init__(self, num_classes, input_dim=3, hidden_dim=256):
        super(HybridSketchNet, self).__init__()
        
        # CNN for rasterized image
        self.cnn = models.resnet18(pretrained=True)
        self.cnn.conv1 = nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)
        self.cnn.fc = nn.Identity()
        cnn_feat_dim = 512
        
        # RNN for stroke sequence
        self.rnn = nn.LSTM(input_dim, hidden_dim, 2, batch_first=True, 
                          bidirectional=True, dropout=0.3)
        rnn_feat_dim = hidden_dim * 2
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(cnn_feat_dim + rnn_feat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, seq, img):
        # CNN features
        cnn_feat = self.cnn(img)
        
        # RNN features
        rnn_out, (h_n, _) = self.rnn(seq)
        rnn_feat = torch.cat([h_n[-2], h_n[-1]], dim=1)
        
        # Fusion
        combined = torch.cat([cnn_feat, rnn_feat], dim=1)
        
        return self.fusion(combined)


def get_recognition_model(model_type, num_classes, **kwargs):
    """Factory function to create recognition models."""
    
    if model_type == 'cnn':
        return SketchCNN(num_classes, **kwargs)
    elif model_type == 'rnn':
        return SketchRNN(num_classes, **kwargs)
    elif model_type == 'transformer':
        return SketchTransformer(num_classes, **kwargs)
    elif model_type == 'mlp':
        return SketchMLP(num_classes, **kwargs)
    elif model_type == 'hybrid':
        return HybridSketchNet(num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == '__main__':
    # Test models
    batch_size = 4
    seq_len = 200
    num_classes = 345
    
    # Test stroke-based models
    seq_input = torch.randn(batch_size, seq_len, 3)
    
    rnn_model = SketchRNN(num_classes)
    out = rnn_model(seq_input)
    print(f"SketchRNN output: {out.shape}")
    
    transformer_model = SketchTransformer(num_classes)
    out = transformer_model(seq_input)
    print(f"SketchTransformer output: {out.shape}")
    
    # Test image-based model
    img_input = torch.randn(batch_size, 1, 224, 224)
    
    cnn_model = SketchCNN(num_classes)
    out = cnn_model(img_input)
    print(f"SketchCNN output: {out.shape}")
    
    # Test hybrid model
    hybrid_model = HybridSketchNet(num_classes)
    out = hybrid_model(seq_input, img_input)
    print(f"HybridSketchNet output: {out.shape}")
