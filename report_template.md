# Sketch Recognition and Generation Project Report

## 项目信息

| 项目 | 内容 |
|------|------|
| 课程 | CS3308 Machine Learning |
| 任务 | Unimodal Task: Sketch Recognition + Sketch Generation |
| 数据集 | QuickDraw-414k (Recognition), QuickDraw npz (Generation) |

---

## 1. 主要思路 (Main Ideas)

本项目完成了Unimodal Task的两个子任务：
1. **Sketch Recognition**: 基于CNN的草图分类
2. **Sketch Generation**: 基于Sketch-RNN VAE的草图生成

### 核心方法
- **Recognition**: 使用预训练ResNet-18对草图图像进行分类
- **Generation**: 使用Sketch-RNN VAE学习草图的潜在表示并生成新草图

---

## 2. 方法与算法 (Methods and Algorithms)

### 2.1 Sketch Recognition

#### 数据表示
将草图转换为224×224的RGB图像，使用ImageNet预训练的归一化参数。

#### 模型架构 - ResNet-18
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

#### 训练策略
- **优化器**: AdamW (lr=1e-3, weight_decay=1e-4)
- **学习率调度**: Cosine Annealing
- **批次大小**: 64
- **训练轮数**: 10

### 2.2 Sketch Generation

#### 数据表示
使用stroke-3格式 [Δx, Δy, pen_state]，序列长度限制为200。

#### 模型架构 - Sketch-RNN VAE
```
Encoder:
  Bidirectional LSTM (hidden=256)
      ↓
  FC → μ, σ (latent_dim=128)
      ↓
  Reparameterization: z = μ + σ * ε

Decoder:
  z → FC → Initial hidden state
      ↓
  LSTM (hidden=512)
      ↓
  GMM Output (20 mixtures): π, μx, μy, σx, σy, ρ, pen_states
```

#### 训练策略
- **损失函数**: Reconstruction Loss (GMM NLL) + KL Divergence
- **KL权重退火**: 从0.01逐渐增加到0.5
- **优化器**: Adam (lr=1e-3)
- **批次大小**: 64
- **训练轮数**: 10

---

## 3. 实验设置 (Experimental Settings)

### 3.1 Sketch Recognition 数据集

| 项目 | 数值 |
|------|------|
| 数据集 | QuickDraw-414k |
| 类别数 | 5 |
| 训练集 | 1,500 samples |
| 验证集 | 500 samples |
| 测试集 | 500 samples |

选择的类别: cat, dog, fish, bird, apple

### 3.2 Sketch Generation 数据集

| 项目 | 数值 |
|------|------|
| 数据集 | QuickDraw npz |
| 类别数 | 5 |
| 训练集 | 10,000 samples |
| 验证集 | 1,000 samples |
| 测试集 | 1,000 samples |

选择的类别: cat, apple, fish, pig, butterfly

### 3.3 评估指标

**Recognition:**
- Accuracy, Precision, Recall, F1-Score

**Generation:**
- Reconstruction Loss (GMM Negative Log-Likelihood)
- KL Divergence
- 生成样本可视化

---

## 4. 实验结果 (Experimental Results)

### 4.1 Sketch Recognition 结果

#### 整体性能

| 指标 | 数值 |
|------|------|
| **Test Accuracy** | **90.40%** |
| Best Val Accuracy | 91.00% |
| Train Accuracy | 96.67% |

#### 各类别性能

| 类别 | Precision | Recall | F1-Score |
|------|-----------|--------|----------|
| cat | 0.913 | 0.840 | 0.875 |
| dog | 0.845 | 0.870 | 0.857 |
| fish | 0.881 | 0.960 | 0.919 |
| bird | 0.907 | 0.880 | 0.893 |
| apple | 0.980 | 0.970 | 0.975 |

#### 训练过程

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|-------|------------|-----------|----------|---------|
| 1 | 0.7751 | 69.20% | 1.5491 | 51.40% |
| 5 | 0.2435 | 90.87% | 0.3313 | 88.60% |
| 10 | 0.0997 | 96.67% | 0.2753 | 91.00% |

### 4.2 Sketch Generation 结果

#### 训练性能

| 指标 | 数值 |
|------|------|
| **Best Val Loss** | **-9.80** |
| Final Recon Loss | -9.93 |
| Final KL Divergence | 0.56 |

#### 训练过程

| Epoch | Train Loss | Recon Loss | KL Loss | Val Loss |
|-------|------------|------------|---------|----------|
| 1 | 0.03 | -0.03 | 5.98 | 0.28 |
| 5 | -3.98 | -4.06 | 0.77 | -5.01 |
| 10 | -9.41 | -9.55 | 0.58 | -9.80 |

### 4.3 可视化

#### Recognition 混淆矩阵
[插入 checkpoints/recognition_results/confusion_matrix.png]

#### Recognition 训练曲线
[插入 checkpoints/recognition_curves.png]

#### Generation 训练曲线
[插入 checkpoints/generation_curves.png]

#### 生成样本
[插入 checkpoints/generated_samples.png]

---

## 5. 讨论 (Discussion)

### 5.1 Sketch Recognition 分析

1. **预训练权重的重要性**: 使用ImageNet预训练的ResNet-18显著提升了分类性能，仅需10个epoch即可达到90%+准确率
2. **类别间差异**: 
   - apple类别表现最好 (F1=0.975)，因为其形状特征明显
   - cat和dog容易混淆 (F1分别为0.875和0.857)，因为都是动物类别
3. **快速收敛**: 模型在第3个epoch就达到了86%的验证准确率

### 5.2 Sketch Generation 分析

1. **VAE训练稳定性**: KL权重退火策略有效避免了KL collapse问题
2. **重建质量**: 负的重建损失表明模型学会了草图的分布
3. **潜在空间**: KL散度稳定在0.56左右，表明潜在空间有良好的结构

### 5.3 局限性

- Recognition: 栅格化过程丢失了笔画顺序信息
- Generation: 生成的草图可能缺乏细节
- 训练数据量有限（每类仅使用部分样本）

### 5.4 改进方向

- 结合Transformer架构处理序列数据
- 使用更大的数据集和更多类别
- 探索条件生成（指定类别生成）

---

## 6. 结论 (Conclusion)

本项目成功完成了Unimodal Task的两个子任务：

1. **Sketch Recognition**: 基于ResNet-18的草图分类模型，在5个类别上达到了**90.40%**的测试准确率
2. **Sketch Generation**: 基于Sketch-RNN VAE的草图生成模型，验证损失达到**-9.80**

实验结果表明：
- 预训练CNN能够有效识别手绘草图
- VAE能够学习草图的潜在表示并生成新样本

---

## 7. 参考文献 (References)

[1] Xu, Peng, et al. "Multigraph transformer for free-hand sketch recognition." IEEE TNNLS, 2021.

[2] Ha, David, and Douglas Eck. "A neural representation of sketch drawings." arXiv:1704.03477, 2017.

[3] Yu, Qian, et al. "Sketch-a-net: A deep neural network that beats humans." IJCV, 2017.

[5] Li, Lei, et al. "Sketch-R2CNN: An RNN-rasterization-CNN architecture." IEEE TVCG, 2020.

[6] Li, Tengjie, et al. "SketchMLP: effectively utilize rasterized images and drawing sequences." ML, 2025.

---

## 8. 贡献说明 (Contribution)

| 姓名 | 学号 | 贡献比例 | 工作内容 |
|------|------|----------|----------|
| A | 00001 | XX% | 模型设计与实现 |
| B | 00002 | XX% | 数据处理与实验 |
| C | 00003 | XX% | 报告撰写与分析 |

---

## 9. 代码仓库

GitHub: https://github.com/YOUR_USERNAME/sketch-recognition

