"""
Sketch Recognition Dataset
使用QuickDraw414k图片数据
"""

import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class QuickDrawImageDataset(Dataset):
    """QuickDraw图片数据集"""
    
    def __init__(self, data_dir, split='train', categories=None, transform=None, max_samples_per_class=None):
        """
        Args:
            data_dir: QuickDraw414k根目录
            split: 'train', 'val', 或 'test'
            categories: 要使用的类别列表
            transform: 图像变换
            max_samples_per_class: 每类最大样本数（用于快速测试）
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        
        # 图片目录 - train/val/test各自独立
        self.img_dir = os.path.join(data_dir, 'picture_files', split)
        
        # 获取所有类别
        if categories is None:
            train_dir = os.path.join(data_dir, 'picture_files', 'train')
            categories = sorted([d for d in os.listdir(train_dir) 
                               if os.path.isdir(os.path.join(train_dir, d))])
        
        self.categories = categories
        self.category_to_idx = {cat: idx for idx, cat in enumerate(categories)}
        
        # 直接从目录加载数据（不使用txt文件）
        self.samples = []
        
        for cat in categories:
            cat_dir = os.path.join(self.img_dir, cat)
            if not os.path.exists(cat_dir):
                continue
            
            files = sorted([f for f in os.listdir(cat_dir) if f.endswith('.png')])
            
            if max_samples_per_class is not None:
                files = files[:max_samples_per_class]
            
            for f in files:
                full_path = os.path.join(cat_dir, f)
                self.samples.append((full_path, self.category_to_idx[cat]))
        
        print(f"Loaded {len(self.samples)} samples for {split}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # 加载图片
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms(img_size=224, is_train=True):
    """获取数据变换"""
    if is_train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])


def create_dataloaders(data_dir, categories=None, batch_size=32, 
                       img_size=224, num_workers=0, max_samples_per_class=None):
    """创建数据加载器"""
    
    train_transform = get_transforms(img_size, is_train=True)
    val_transform = get_transforms(img_size, is_train=False)
    
    train_dataset = QuickDrawImageDataset(
        data_dir, 'train', categories, train_transform, max_samples_per_class
    )
    val_dataset = QuickDrawImageDataset(
        data_dir, 'val', categories, val_transform, max_samples_per_class
    )
    test_dataset = QuickDrawImageDataset(
        data_dir, 'test', categories, val_transform, max_samples_per_class
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader, train_dataset.categories


if __name__ == '__main__':
    # 测试
    data_dir = 'D:/AAAAAAAAAAAAAproject/QuickDraw414k'
    categories = ['cat', 'dog', 'fish', 'bird', 'apple']
    
    train_loader, val_loader, test_loader, cats = create_dataloaders(
        data_dir, categories, batch_size=32, max_samples_per_class=100
    )
    
    print(f"Categories: {cats}")
    print(f"Train batches: {len(train_loader)}")
    
    batch = next(iter(train_loader))
    print(f"Batch shape: {batch[0].shape}")
