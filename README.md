# Sketch Recognition and Generation

CS3308 Machine Learning - Unimodal Task

基于深度学习的草图识别与生成系统。

## 项目概述

本项目完成了Unimodal Task的两个子任务：
- **Sketch Recognition**: 基于ResNet-18的草图分类 (Test Accuracy: **90.40%**)
- **Sketch Generation**: 基于Sketch-RNN VAE的草图生成 (Val Loss: **-9.80**)

## 项目结构

```
sketch_project/
├── recognition/                    # 草图识别模块
│   ├── dataset.py                  # QuickDraw图片数据加载
│   ├── model.py                    # ResNet-18分类模型
│   ├── train.py                    # 训练脚本
│   ├── evaluate.py                 # 评估脚本
│   ├── main.py                     # 主程序
│   └── checkpoints/                # 模型和结果
│       ├── best_recognition_model.pth
│       ├── recognition_curves.png
│       └── recognition_results/
│
├── sketch_generation/              # 草图生成模块
│   ├── dataset.py                  # QuickDraw npz数据加载
│   ├── model.py                    # Sketch-RNN VAE模型
│   ├── train.py                    # 训练脚本
│   ├── evaluate.py                 # 评估脚本
│   ├── main.py                     # 主程序
│   └── checkpoints/                # 模型和结果
│       ├── best_generation_model.pth
│       ├── generation_training_curves.png
│       └── generated_samples.png
│
├── report_template.md              # 项目报告
├── requirements.txt                # 依赖
└── README.md                       # 说明文档
```

## 环境配置

```bash
pip install -r requirements.txt
```

主要依赖：
- Python 3.8+
- PyTorch 1.10+
- torchvision
- numpy
- matplotlib
- scikit-learn

## 数据集准备

### Sketch Recognition
1. 下载 QuickDraw-414k 数据集
2. 解压到 `D:/AAAAAAAAAAAAAproject/QuickDraw414k/`
3. 确保目录结构：`picture_files/train/类别名/*.png`

### Sketch Generation
1. 下载 QuickDraw npz 文件
2. 放置到 `sketch_generation_data/` 目录
3. 文件格式：`类别名.npz`

## 快速开始

### Sketch Recognition

```bash
cd recognition

# 训练+评估
python main.py --mode all --epochs 10 --categories cat dog fish bird apple

# 仅训练
python main.py --mode train --epochs 10 --max_samples 300

# 仅评估
python main.py --mode evaluate
```

### Sketch Generation

```bash
cd sketch_generation

# 训练
python main.py --mode train --epochs 10 --max_samples 2000 --categories cat apple fish pig butterfly

# 生成样本
python main.py --mode generate

# 完整流程
python main.py --mode all
```

## 参数说明

### Recognition 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data_dir` | QuickDraw414k路径 | 数据集目录 |
| `--categories` | 10类 | 选择的类别 |
| `--epochs` | 15 | 训练轮数 |
| `--batch_size` | 64 | 批次大小 |
| `--lr` | 1e-3 | 学习率 |
| `--max_samples` | 500 | 每类最大样本数 |
| `--backbone` | resnet18 | 骨干网络 |

### Generation 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data_dir` | sketch_generation_data | 数据集目录 |
| `--categories` | 5类 | 选择的类别 |
| `--epochs` | 50 | 训练轮数 |
| `--batch_size` | 64 | 批次大小 |
| `--latent_dim` | 128 | 潜在空间维度 |
| `--max_samples` | 2000 | 每类最大样本数 |

## 模型架构

### Sketch Recognition - ResNet-18
```
Input (3, 224, 224)
    ↓
ResNet-18 Backbone (ImageNet pretrained)
    ↓
Global Average Pooling
    ↓
Dropout (0.5) → FC (256) → ReLU
    ↓
Dropout (0.25) → FC (num_classes)
    ↓
Output (class probabilities)
```

### Sketch Generation - Sketch-RNN VAE
```
Encoder: Bidirectional LSTM → μ, σ → z
Decoder: z → LSTM → GMM (20 mixtures) → [Δx, Δy, pen_state]
```

## 实验结果

### Recognition Results

| 类别 | Precision | Recall | F1-Score |
|------|-----------|--------|----------|
| cat | 0.913 | 0.840 | 0.875 |
| dog | 0.845 | 0.870 | 0.857 |
| fish | 0.881 | 0.960 | 0.919 |
| bird | 0.907 | 0.880 | 0.893 |
| apple | 0.980 | 0.970 | 0.975 |
| **Overall** | - | - | **90.40%** |

### Generation Results

| 指标 | 数值 |
|------|------|
| Best Val Loss | -9.80 |
| Final Recon Loss | -9.93 |
| Final KL Divergence | 0.56 |

## 输出文件

### Recognition
- `checkpoints/best_recognition_model.pth` - 最佳模型
- `checkpoints/recognition_curves.png` - 训练曲线
- `checkpoints/recognition_results/confusion_matrix.png` - 混淆矩阵

### Generation
- `checkpoints/best_generation_model.pth` - 最佳模型
- `checkpoints/generation_training_curves.png` - 训练曲线
- `checkpoints/generated_samples.png` - 生成样本

## 参考文献

[1] Xu, Peng, et al. "Multigraph transformer for free-hand sketch recognition." IEEE TNNLS, 2021.

[2] Ha, David, and Douglas Eck. "A neural representation of sketch drawings." arXiv:1704.03477, 2017.

[3] Yu, Qian, et al. "Sketch-a-net: A deep neural network that beats humans." IJCV, 2017.

## GitHub

项目代码: https://github.com/hua691/sketch-recognition-generation
