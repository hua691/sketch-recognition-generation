"""
Sketch Generation Dataset
加载QuickDraw npz格式数据，用于Sketch-RNN训练
参考: https://github.com/CMACH508/SketchEdit
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def load_npz_data(data_dir, categories=None, max_seq_len=200, max_samples_per_class=5000):
    """
    加载npz格式的QuickDraw数据
    
    Args:
        data_dir: 包含.npz文件的目录
        categories: 要加载的类别列表，None表示全部
        max_seq_len: 最大序列长度
        max_samples_per_class: 每类最大样本数（用于快速训练）
    
    Returns:
        train_data, val_data, test_data: 字典，包含sketches和labels
    """
    train_sketches, val_sketches, test_sketches = [], [], []
    train_labels, val_labels, test_labels = [], [], []
    
    # 获取所有npz文件
    if categories is None:
        npz_files = [f for f in os.listdir(data_dir) if f.endswith('.npz')]
        categories = [f.replace('.npz', '') for f in npz_files]
    
    category_to_idx = {cat: idx for idx, cat in enumerate(categories)}
    
    for cat in categories:
        npz_path = os.path.join(data_dir, f'{cat}.npz')
        if not os.path.exists(npz_path):
            print(f"Warning: {npz_path} not found, skipping...")
            continue
        
        print(f"Loading {cat}...")
        data = np.load(npz_path, allow_pickle=True, encoding='latin1')
        
        label = category_to_idx[cat]
        
        # 加载训练/验证/测试数据（限制每类样本数）
        if 'train' in data:
            count = 0
            for sketch in data['train']:
                if max_samples_per_class and count >= max_samples_per_class:
                    break
                sketch = preprocess_sketch(sketch, max_seq_len)
                if sketch is not None:
                    train_sketches.append(sketch)
                    train_labels.append(label)
                    count += 1
        
        if 'valid' in data:
            count = 0
            for sketch in data['valid']:
                if max_samples_per_class and count >= max_samples_per_class // 10:
                    break
                sketch = preprocess_sketch(sketch, max_seq_len)
                if sketch is not None:
                    val_sketches.append(sketch)
                    val_labels.append(label)
                    count += 1
        
        if 'test' in data:
            count = 0
            for sketch in data['test']:
                if max_samples_per_class and count >= max_samples_per_class // 10:
                    break
                sketch = preprocess_sketch(sketch, max_seq_len)
                if sketch is not None:
                    test_sketches.append(sketch)
                    test_labels.append(label)
                    count += 1
    
    print(f"Loaded: train={len(train_sketches)}, val={len(val_sketches)}, test={len(test_sketches)}")
    
    return {
        'train': {'sketches': train_sketches, 'labels': train_labels},
        'val': {'sketches': val_sketches, 'labels': val_labels},
        'test': {'sketches': test_sketches, 'labels': test_labels},
        'categories': categories
    }


def preprocess_sketch(sketch, max_seq_len=200):
    """
    预处理草图序列
    - 格式: [dx, dy, pen_state] 或 [dx, dy, p1, p2, p3]
    - 归一化坐标
    - 截断或填充到固定长度
    """
    if len(sketch) == 0:
        return None
    
    # 确保是stroke-3格式 [dx, dy, pen_state]
    if sketch.shape[1] == 5:
        # stroke-5 转 stroke-3
        stroke3 = np.zeros((len(sketch), 3))
        stroke3[:, :2] = sketch[:, :2]
        stroke3[:, 2] = sketch[:, 3] + sketch[:, 4]  # pen up or end
        sketch = stroke3
    elif sketch.shape[1] != 3:
        return None
    
    # 归一化坐标
    sketch = normalize_sketch(sketch)
    
    # 截断
    if len(sketch) > max_seq_len:
        sketch = sketch[:max_seq_len]
    
    return sketch.astype(np.float32)


def normalize_sketch(sketch, scale_factor=None):
    """归一化草图坐标到[-1, 1]范围"""
    sketch = sketch.copy()
    
    # 计算绝对坐标
    abs_coords = np.cumsum(sketch[:, :2], axis=0)
    
    # 计算缩放因子
    if scale_factor is None:
        min_val = abs_coords.min()
        max_val = abs_coords.max()
        scale_factor = max(1.0, max_val - min_val)
    
    # 归一化delta
    sketch[:, :2] = sketch[:, :2] / scale_factor
    
    return sketch


def pad_sketch(sketch, max_len, pad_value=0):
    """填充草图到固定长度"""
    if len(sketch) >= max_len:
        return sketch[:max_len]
    
    padding = np.zeros((max_len - len(sketch), sketch.shape[1]))
    padding[:, 2] = 1  # pen up for padding
    
    return np.vstack([sketch, padding])


class SketchDataset(Dataset):
    """Sketch Generation数据集"""
    
    def __init__(self, sketches, labels, max_seq_len=200):
        self.sketches = sketches
        self.labels = labels
        self.max_seq_len = max_seq_len
    
    def __len__(self):
        return len(self.sketches)
    
    def __getitem__(self, idx):
        sketch = self.sketches[idx].copy()
        label = self.labels[idx]
        
        # 填充到固定长度
        sketch = pad_sketch(sketch, self.max_seq_len)
        
        # 转换为tensor
        sketch = torch.tensor(sketch, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        
        # 创建mask (1表示有效位置)
        seq_len = min(len(self.sketches[idx]), self.max_seq_len)
        mask = torch.zeros(self.max_seq_len)
        mask[:seq_len] = 1
        
        return {
            'sketch': sketch,
            'label': label,
            'mask': mask,
            'seq_len': seq_len
        }


def create_dataloaders(data_dir, categories=None, batch_size=64, 
                       max_seq_len=200, num_workers=0, max_samples_per_class=5000):
    """创建数据加载器"""
    
    data = load_npz_data(data_dir, categories, max_seq_len, max_samples_per_class)
    
    train_dataset = SketchDataset(
        data['train']['sketches'], 
        data['train']['labels'],
        max_seq_len
    )
    val_dataset = SketchDataset(
        data['val']['sketches'],
        data['val']['labels'],
        max_seq_len
    )
    test_dataset = SketchDataset(
        data['test']['sketches'],
        data['test']['labels'],
        max_seq_len
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                              shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader, data['categories']


if __name__ == '__main__':
    # 测试数据加载
    data_dir = '../sketch_generation_data'
    train_loader, val_loader, test_loader, categories = create_dataloaders(
        data_dir, batch_size=32, max_seq_len=200
    )
    
    print(f"Categories: {categories}")
    print(f"Train batches: {len(train_loader)}")
    
    # 测试一个batch
    batch = next(iter(train_loader))
    print(f"Sketch shape: {batch['sketch'].shape}")
    print(f"Label shape: {batch['label'].shape}")
