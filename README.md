# Sketch Recognition - Task 2A

基于深度学习的草图识别系统，使用QuickDraw-414k数据集。

## 项目结构

```
sketch_project/
├── data/
│   └── quickdraw_dataset.py    # 数据集加载
├── models/
│   └── sketch_recognition.py   # 识别模型
├── train_recognition.py        # 训练脚本
├── evaluate_sketch.py          # 评估脚本
├── main.py                     # 主脚本
├── requirements.txt            # 依赖
└── README.md                   # 说明文档
```

## 环境配置

```bash
pip install -r requirements.txt
```

## 数据集准备

1. 从云存储下载 QuickDraw-414k 数据集
2. 解压到 `./quickdraw_data` 目录
3. 数据格式支持 `.npz` 或 `.ndjson`

数据加载参考：
- https://github.com/PengBoXiangShang/torchsketch/tree/master/torchsketch/data/dataloaders/quickdraw/quickdraw_414k

## 快速开始

### 训练+评估
```bash
python main.py --mode all --model cnn --categories cat dog car house tree
```

### 仅训练
```bash
python main.py --mode train --epochs 50 --batch_size 64
```

### 仅评估
```bash
python main.py --mode evaluate
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | cnn | 模型类型: cnn, rnn, transformer |
| `--representation` | image | 数据表示: image (栅格化), stroke3 (序列) |
| `--categories` | 10类 | 选择5-10个类别 |
| `--epochs` | 50 | 训练轮数 |
| `--batch_size` | 64 | 批次大小 |
| `--lr` | 1e-4 | 学习率 |

## 模型架构

### SketchCNN (推荐)
- 基于ResNet的CNN分类器
- 输入：栅格化草图图像 (224x224)
- 使用ImageNet预训练权重

### SketchRNN
- 双向LSTM + 注意力机制
- 输入：stroke-3序列 [dx, dy, pen_state]

### SketchTransformer
- Transformer编码器
- 输入：stroke-3序列

## 评估指标

- **Accuracy**: 分类准确率
- **Precision/Recall/F1**: 加权平均
- **Confusion Matrix**: 混淆矩阵可视化
- **95% CI**: Bootstrap置信区间

## 输出文件

```
checkpoints/
├── best_recognition_model.pth  # 最佳模型
├── recognition_history.json    # 训练历史
├── recognition_curves.png      # 训练曲线
└── results/
    ├── recognition_results.json  # 评估结果
    └── confusion_matrix.png      # 混淆矩阵
```

## 参考文献

[1] Xu, Peng, et al. "Multigraph transformer for free-hand sketch recognition." IEEE TNNLS, 2021.
[3] Yu, Qian, et al. "Sketch-a-net: A deep neural network that beats humans." IJCV, 2017.
[5] Li, Lei, et al. "Sketch-R2CNN: An RNN-rasterization-CNN architecture." IEEE TVCG, 2020.
[6] Li, Tengjie, et al. "SketchMLP: effectively utilize rasterized images and drawing sequences." ML, 2025.

## 贡献说明

| 姓名 | 学号 | 贡献比例 | 工作内容 |
|------|------|----------|----------|
| A | 00001 | XX% | - |
| B | 00002 | XX% | - |
| C | 00003 | XX% | - |

## GitHub

项目代码: [Your GitHub Link]
