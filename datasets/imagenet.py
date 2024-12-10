import os
import pandas as pd
import torch
from torchvision import datasets, transforms
from typing import Dict, Tuple, Optional
from PIL import Image

transforms_ = {
    'train': [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
    ],
    'val': [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
    ]
}

def load_class_mappings(root: str) -> Tuple[Dict[str, int], Dict[str, str]]:
    """
    Load class mappings from wnids.txt and words.txt
    
    Returns:
        wnid_to_idx: mapping from WordNet ID to class index
        wnid_to_name: mapping from WordNet ID to class name
    """
    # Load WordNet IDs and create index mapping
    with open(os.path.join(root, 'wnids.txt'), 'r') as f:
        wnids = [x.strip() for x in f]
    wnid_to_idx = {wnid: i for i, wnid in enumerate(wnids)}
    
    # Load class descriptions
    wnid_to_name = {}
    with open(os.path.join(root, 'words.txt'), 'r') as f:
        for line in f:
            wnid, name = line.strip().split('\t', 1)
            wnid_to_name[wnid] = name
            
    return wnid_to_idx, wnid_to_name

# def get_tinyimagenet(data_path: str, split: str = "train", preprocessing: bool = True):
def get_imagenet(data_path: str, split: str = "train", preprocessing: bool = True):
    """
    Get TinyImageNet dataset loader
    
    Args:
        data_path: Root directory of TinyImageNet dataset
        split: 'train', 'val', or 'test'
        preprocessing: If True, apply normalization
    """
    if split == "train":
        transform = transforms_['train']
    else:
        transform = transforms_['val']
        
    if not preprocessing:
        transform = transforms.Compose([t for t in transform if not isinstance(t, transforms.Normalize)])
    else:
        transform = transforms.Compose(transform)
        
#     return TinyImageNetDataset(data_path, split=split, transform=transform)
    return ImageNetDataset(data_path, split=split, transform=transform)

# class TinyImageNetDataset(torch.utils.data.Dataset):
class ImageNetDataset(torch.utils.data.Dataset):
    def __init__(self, root: str, split: str = 'train', transform: Optional[transforms.Compose] = None):
        """
        TinyImageNet Dataset
        
        Args:
            root: Root directory of TinyImageNet dataset
            split: 'train', 'val', or 'test'
            transform: Optional transform to be applied on images
        """
        self.root = root
        self.split = split
        self.transform = transform
        
        # Load class mappings
        self.wnid_to_idx, self.wnid_to_name = load_class_mappings(root)
        self.classes = list(self.wnid_to_idx.keys())
        self.num_classes = len(self.classes)
        
        # Setup data paths and load annotations
        self.samples = []  # Will contain (image_path, class_idx) tuples
        self.boxes = {}    # Will contain bounding boxes if available
        
        if split == 'train':
            # For training, traverse class directories
            for wnid in self.classes:
                class_dir = os.path.join(root, 'train', wnid)
                if not os.path.isdir(class_dir):
                    continue
                    
                # Load bounding boxes
                box_file = os.path.join(class_dir, f'{wnid}_boxes.txt')
                if os.path.exists(box_file):
                    with open(box_file, 'r') as f:
                        for line in f:
                            fname, *coords = line.strip().split()
                            img_path = os.path.join(class_dir, 'images', fname)
                            self.boxes[img_path] = list(map(int, coords))
                            self.samples.append((img_path, self.wnid_to_idx[wnid]))
                            
        elif split == 'val':
            # Load validation annotations
            val_anno_path = os.path.join(root, 'val', 'val_annotations.txt')
            with open(val_anno_path, 'r') as f:
                for line in f:
                    fname, wnid, *coords = line.strip().split()
                    img_path = os.path.join(root, 'val', 'images', fname)
                    if os.path.exists(img_path):
                        self.boxes[img_path] = list(map(int, coords))
                        self.samples.append((img_path, self.wnid_to_idx[wnid]))
                        
        else:  # test
            # For test, we just load images without labels
            test_dir = os.path.join(root, 'test', "images")
            if os.path.exists(test_dir):
                for fname in sorted(os.listdir(test_dir)):
                    if fname.endswith(('.JPEG', '.jpeg', '.jpg')):
                        img_path = os.path.join(test_dir, fname)
                        self.samples.append((img_path, -1))  # Use -1 as class_idx for test
        
        self.preprocessing = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        print(f"Dataset length:{len(self)}, split: {self.split}")

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """
        Args:
            index: Index
            
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        
        # Load image
        img = Image.open(path).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)
            
        return img, target
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def get_box(self, index: int) -> Optional[list]:
        """Get bounding box coordinates for an image"""
        path, _ = self.samples[index]
        return self.boxes.get(path)

    def reverse_augmentation(self, data: torch.Tensor) -> torch.Tensor:
        """Reverse the normalization transform"""
        data = data.clone()
        mean = torch.Tensor(self.preprocessing.mean).to(data)
        var = torch.Tensor(self.preprocessing.std).to(data)
        data *= var[:, None, None]
        data += mean[:, None, None]
        return torch.multiply(data, 255)
    
    def get_class_name(self, class_idx: int) -> str:
        """Get class name from class index"""
        wnid = self.classes[class_idx]
        return self.wnid_to_name.get(wnid, wnid)