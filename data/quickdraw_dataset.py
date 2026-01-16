"""QuickDraw-414k Dataset Loader
Supports QuickDraw414k image folder structure
Reference: https://github.com/PengBoXiangShang/torchsketch
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms


class QuickDrawDataset(Dataset):
    """QuickDraw-414k dataset for sketch recognition."""
    
    def __init__(self, data_dir, categories=None, split='train', 
                 img_size=224, transform=None):
        """
        Args:
            data_dir: Root directory (e.g., quickdraw_data/QuickDraw414k)
            categories: List of category names to load (None = all available)
            split: 'train', 'val', or 'test'
            img_size: Image size for resizing
            transform: Optional transform for images
        """
        self.data_dir = data_dir
        self.split = split
        self.img_size = img_size
        self.transform = transform
        
        self.image_paths = []
        self.labels = []
        
        # Determine picture_files path
        self.picture_dir = os.path.join(data_dir, 'picture_files')
        if not os.path.exists(self.picture_dir):
            self.picture_dir = data_dir
        
        # Find available categories
        self.categories = categories if categories else self._find_categories()
        self.category_to_idx = {cat: idx for idx, cat in enumerate(self.categories)}
        
        self._load_data()
    
    def _find_categories(self):
        """Find all available categories in train directory."""
        train_dir = os.path.join(self.picture_dir, 'train')
        if os.path.exists(train_dir):
            return sorted([d for d in os.listdir(train_dir) 
                          if os.path.isdir(os.path.join(train_dir, d))])
        return []
    
    def _load_data(self):
        """Load image paths from directory structure."""
        # Use txt files if available, otherwise use folder structure
        if self.split == 'train':
            txt_file = os.path.join(self.picture_dir, 'tiny_train_set.txt')
            img_dir = os.path.join(self.picture_dir, 'train')
        elif self.split == 'val':
            txt_file = os.path.join(self.picture_dir, 'tiny_val_set.txt')
            img_dir = os.path.join(self.picture_dir, 'train')  # val uses train images
        else:
            txt_file = os.path.join(self.picture_dir, 'tiny_test_set.txt')
            img_dir = os.path.join(self.picture_dir, 'test')
        
        if os.path.exists(txt_file):
            self._load_from_txt(txt_file, img_dir)
        else:
            self._load_from_folders(img_dir)
        
        print(f"Loaded {len(self.image_paths)} images from {len(self.categories)} categories for {self.split}")
    
    def _load_from_txt(self, txt_file, img_dir):
        """Load from txt file (format: category/filename.png label_id)."""
        with open(txt_file, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 1:
                rel_path = parts[0]  # e.g., "banana/banana_0.png"
                category = rel_path.split('/')[0]
                
                # Only load specified categories
                if category in self.categories:
                    img_path = os.path.join(img_dir, rel_path)
                    if os.path.exists(img_path):
                        self.image_paths.append(img_path)
                        self.labels.append(self.category_to_idx[category])
    
    def _load_from_folders(self, img_dir):
        """Load from folder structure (category/images)."""
        for category in self.categories:
            cat_dir = os.path.join(img_dir, category)
            if os.path.exists(cat_dir):
                label = self.category_to_idx[category]
                for img_name in os.listdir(cat_dir):
                    if img_name.endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(cat_dir, img_name)
                        self.image_paths.append(img_path)
                        self.labels.append(label)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        else:
            # Default transform
            img = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
            ])(img)
        
        return img, label


def create_dataloaders(data_dir, categories, batch_size=64, img_size=224, num_workers=0):
    """Create train/val/test data loaders."""
    
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = QuickDrawDataset(
        data_dir, categories, 'train', img_size, train_transform
    )
    val_dataset = QuickDrawDataset(
        data_dir, categories, 'val', img_size, test_transform
    )
    test_dataset = QuickDrawDataset(
        data_dir, categories, 'test', img_size, test_transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers)
    
    return train_loader, val_loader, test_loader, train_dataset.categories


if __name__ == '__main__':
    # Test dataset
    data_dir = './quickdraw_data/QuickDraw414k'
    categories = ['apple', 'banana', 'airplane']
    
    dataset = QuickDrawDataset(data_dir, categories, 'train')
    print(f"Dataset size: {len(dataset)}")
    print(f"Categories: {dataset.categories}")
    
    if len(dataset) > 0:
        sample, label = dataset[0]
        print(f"Sample shape: {sample.shape}, Label: {label}")
