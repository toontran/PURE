import os
from argparse import ArgumentParser
import random
import numpy as np

from crp.attribution import CondAttribution
from crp.concepts import ChannelConcept
from crp.image import *
from crp.visualization import FeatureVisualization
from zennit.composites import EpsilonPlusFlat

from datasets import get_dataset
from models import get_fn_model_loader, get_canonizer
from models.timm_resnet_detection import get_resnet_timm, get_resnet50_timm, get_resnet34_timm, get_resnet101_timm, get_resnet_canonizer
from utils.helper import load_config, get_layer_names_model
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import pandas as pd
import numpy as np
import re
from sklearn.utils import resample
from pathlib import Path
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

database_path = '/home/ttran02/.cache/kagglehub/datasets/mbornoe/lisa-traffic-light-dataset/versions/2/'
target_classes = ['go', 'stop', 'warning','background']
color_map = {'go':'green', 'stop':'red', 'warning':'yellow','background':'gray'}
# rgb_color_map = {'go': (0, 255, 0), 'stop': (255, 0, 0), 'warning': (255, 255, 0)}

train_folder_list = [
    'dayTrain',
#     'daySequence1',
#     'daySequence2',
#     'sample-dayClip6',
#     'nightTrain',
#     'nightSequence1',
#     'nightSequence2',
#     'sample-nightClip1',
]

n_samples_per_class = 5000


MODELS = {
    "resnet50_timm": get_resnet50_timm,
    "resnet34_timm": get_resnet34_timm,
    "resnet101_timm": get_resnet101_timm,
}

CANONIZERS = {
    "resnet50_timm": get_resnet_canonizer,
    "resnet34_timm": get_resnet_canonizer,
    "resnet101_timm": get_resnet_canonizer,
}


def get_canonizer(model_name):
    assert model_name in list(CANONIZERS.keys()), f"No canonizer for model '{model_name}' available"
    return [CANONIZERS[model_name]()]


def get_fn_model_loader(model_name: str) -> torch.nn.Module:
    if model_name in MODELS:
        fn_model_loader = MODELS[model_name]
        return fn_model_loader
    else:
        raise KeyError(f"Model {model_name} not available")

def split_dataframe(df, test_size=0.2, random_state=42):
    """
    Split a dataframe into train and test sets
    
    Args:
        df (pd.DataFrame): Input dataframe
        test_size (float): Proportion of dataset to include in the test split
        random_state (int): Random state for reproducibility
    
    Returns:
        train_df, test_df (tuple of pd.DataFrame)
    """
    
    # Get indices for train and test sets
    train_idx, test_idx = train_test_split(
        df.index,
        test_size=test_size,
        random_state=random_state,
        shuffle=True
    )
    
    # Split the dataframe
    train_df = df.loc[train_idx].reset_index(drop=True)
    test_df = df.loc[test_idx].reset_index(drop=True)
    
    # Print split information
    print(f"Total samples: {len(df)}")
    print(f"Training samples: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"Test samples: {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
    
    return train_df, test_df

def get_annotarion_dataframe(train_data_folders):
    data_base_path = '/home/ttran02/.cache/kagglehub/datasets/mbornoe/lisa-traffic-light-dataset/versions/2/'
    annotation_list = list()
    for folder in [folder + '/' for folder in train_data_folders if os.listdir(data_base_path)]:
        annotation_path = ''
        if 'sample' not in folder:
            annotation_path = data_base_path + 'Annotations/Annotations/' + folder
        else:
            annotation_path = data_base_path + folder*2
        image_frame_path = data_base_path + folder*2
        
        df = pd.DataFrame()
        if 'Clip' in os.listdir(annotation_path)[0]:
            clip_list = os.listdir(annotation_path)
            for clip_folder in clip_list:
                df = pd.read_csv(annotation_path + clip_folder +  '/frameAnnotationsBOX.csv', sep=";")
                df['image_path'] = image_frame_path + clip_folder + '/frames/'
                annotation_list.append(df)
        else:
            df = pd.read_csv(annotation_path +  'frameAnnotationsBOX.csv', sep=";")
            df['image_path'] = image_frame_path + 'frames/'
            annotation_list.append(df)
        
    df = pd.concat(annotation_list)
    df = df.drop(['Origin file', 'Origin frame number', 'Origin track', 'Origin track frame number'], axis=1)
    df.columns = ['filename', 'target', 'x1', 'y1', 'x2', 'y2', 'image_path']
    df = df[df['target'].isin(target_classes)]
    df['filename'] = df['filename'].apply(lambda filename: re.findall("\/([\d\w-]*.jpg)", filename)[0])
    df = df.drop_duplicates().reset_index(drop=True)
    return df

import torch
from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image
import re
from pathlib import Path
import numpy as np
import torchvision.transforms 
import torchvision.transforms.functional 

# class TrafficLightDataset(Dataset):
#     def __init__(self, df, img_dir, transform=None):
#         """
#         Args:
#             df: DataFrame with traffic light annotations
#             img_dir: Directory containing all images
#             transform: Optional transforms to be applied on images
#             include_negatives: Whether to include frames without traffic lights
#         """
#         self.df = df
#         self.img_dir = Path(img_dir)
#         self.transform = transform
        
#         # Group annotations by filename
#         self.grouped_annotations = df.groupby('filename')
        
#         # Get all unique filenames with annotations
#         self.frames = list(set(df['filename'].unique()))
        
#     def _get_boxes_and_labels(self, filename):
# #         filename = filename.split('/')[-1]
        
#         """Get all bounding boxes and labels for a given image"""
# #         if filename not in self.positive_frames:
# #             return torch.zeros((0, 4)), torch.zeros(0, dtype=torch.long)
        
#         annotations = self.grouped_annotations.get_group(filename)
#         boxes = torch.tensor([[row.x1, row.y1, row.x2, row.y2] 
#                             for _, row in annotations.iterrows()], dtype=torch.float32)
        
#         # Convert string labels to integers
#         label_map = {'background': 0, 'stop': 1, 'go': 2, 'warning': 3}
#         labels = torch.tensor([label_map[label] for label in annotations['target']], 
#                             dtype=torch.long)
        
#         return boxes, labels
    
#     def __len__(self):
#         return len(self.frames)
    
#     def __getitem__(self, idx):
#         filename = self.frames[idx]
        
# #         print(filename)
# #         import pdb;pdb.set_trace()
        
#         # Load image
#         clip_name = filename.split('--')[0]
#         img_path = self.img_dir / clip_name / "frames" / filename
#         image = Image.open(img_path).convert('RGB')
        
#         if self.transform:
#             image = self.transform(image)
        
#         # Get boxes and labels
#         boxes, labels = self._get_boxes_and_labels(filename)
        
#         # Create target dictionary
#         target = {
#             'boxes': boxes,
#             'labels': labels,
#             'image_id': torch.tensor([idx]),
#             'area': (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
#             'iscrowd': torch.zeros((len(boxes),), dtype=torch.int64),
#             'filename': filename,
#         }
        
#         return image, target
    

class TrafficLightDataset(Dataset):
    def __init__(self, df, img_dir, target_size=(64,64), transform=None, 
                 scaling_factor=4): #14
        """
        Args:
            df: DataFrame with traffic light annotations
            img_dir: Directory containing all images
            target_size: Tuple of (height, width) for resizing
            transform: Optional additional transforms to be applied after resizing
        """
        self.df = df
        self.img_dir = Path(img_dir)
        self.target_size = target_size 
        self.scaling_factor = scaling_factor
        self.additional_transform = transform
        
        # Group annotations by filename
        self.grouped_annotations = df.groupby('filename')
        
        # Get all unique filenames with annotations
        self.frames = list(set(df['filename'].unique()))
        
        # Label mapping
        self.label_map = {'background': 0, 'stop': 1, 'go': 2, 'warning': 3}
        
    def _resize_image_and_boxes(self, image, boxes):
        """
        Resize image and adjust bounding boxes accordingly
        """
        # Get original size
        orig_w, orig_h = image.size
#         print("sf, w, h", self.scaling_factor, orig_w, orig_h)
        target_h, target_w = (orig_w // self.scaling_factor, orig_h // self.scaling_factor) \
                                    if self.scaling_factor else self.target_size
        self.target_size = target_h, target_w
#         print("w, h", target_h, target_w)
        
        # Compute scaling factors
        w_scale = target_w / orig_w
        h_scale = target_h / orig_h
        
        # Resize image
        image = torchvision.transforms.functional.resize(image, self.target_size)
        
        if len(boxes):
            # Scale bounding boxes
            scaled_boxes = boxes.clone()
            scaled_boxes[:, [0, 2]] *= w_scale  # scale x coordinates
            scaled_boxes[:, [1, 3]] *= h_scale  # scale y coordinates
            
            # Clamp boxes to image boundaries
            scaled_boxes[:, [0, 2]] = torch.clamp(scaled_boxes[:, [0, 2]], 0, target_w)
            scaled_boxes[:, [1, 3]] = torch.clamp(scaled_boxes[:, [1, 3]], 0, target_h)
            
            return image, scaled_boxes
        
        return image, boxes
    
    def _get_boxes_and_labels(self, filename):
        """Get all bounding boxes and labels for a given image"""
        annotations = self.grouped_annotations.get_group(filename)
        boxes = torch.tensor([[row.x1, row.y1, row.x2, row.y2] 
                            for _, row in annotations.iterrows()], dtype=torch.float32)
        
        labels = torch.tensor([self.label_map[label] for label in annotations['target']], 
                            dtype=torch.long)
        
        return boxes, labels
    
    def _validate_boxes(self, boxes):
        """Remove invalid boxes (those with zero width or height after resizing)"""
        if len(boxes) == 0:
            return boxes, torch.zeros(0, dtype=torch.long)
        
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        valid_boxes = (widths > 1) & (heights > 1)
        
        return boxes[valid_boxes], valid_boxes
    
    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, idx):
        filename = self.frames[idx]
        
        # Load image
        clip_name = filename.split('--')[0]
        img_path = self.img_dir / clip_name / "frames" / filename
        image = Image.open(img_path).convert('RGB')
        
        # Get original boxes and labels
        boxes, labels = self._get_boxes_and_labels(filename)
        
        # Resize image and adjust boxes
        image, boxes = self._resize_image_and_boxes(image, boxes)
        
        # Validate boxes after resizing
        boxes, valid_indices = self._validate_boxes(boxes)
        if len(valid_indices) < len(labels):
            labels = labels[valid_indices]
        
        # Apply any additional transforms
        if self.additional_transform:
            image = self.additional_transform(image)
        
        # Convert PIL to tensor if not done by transforms
        if isinstance(image, Image.Image):
            image = torchvision.transforms.functional.to_tensor(image)
        
        # Create target dictionary
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx]),
            'area': (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
            'iscrowd': torch.zeros((len(boxes),), dtype=torch.int64),
            'filename': filename,
        }
        
        return image, target

# Example usage:
def create_traffic_light_data_loader(
    df, img_dir, batch_size=4, transform=None, include_negatives=True, num_workers=4
):
    """
    Create data loader for traffic light detection
    """
    from torch.utils.data import DataLoader
    
    dataset = TrafficLightDataset(
        df=df,
        img_dir=img_dir,
        transform=transform
    )
    
    # Custom collate function to handle variable number of objects
    def collate_fn(batch):
        return tuple(zip(*batch))
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn
    )

# Example transforms
def get_transforms(train=True):
    from torchvision import transforms
    
    if train:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(0.5),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])

class DetectionLoss(nn.Module):
    def __init__(self, num_classes=4, background_weight=0.1, gamma=2.0):
        super().__init__()
        self.num_classes = num_classes
        
        # Initialize focal loss for classification
        alpha = torch.ones(num_classes)
        alpha[0] = background_weight  # Set background weight
        self.register_buffer('alpha', alpha)
        self.gamma = gamma
        
        # Initialize smooth L1 loss for bbox regression
        self.bbox_loss = nn.SmoothL1Loss(reduction='mean')
        
    def forward(self, predictions, targets):
        """
        Args:
            predictions: tuple of (cls_scores, bbox_preds)
                cls_scores: (batch_size, num_anchors, num_classes + 1, H, W)
                bbox_preds: (batch_size, num_anchors, 4, H, W)
            targets: list of dicts, each containing:
                - boxes: tensor (num_boxes, 4)
                - labels: tensor (num_boxes,)
        """
        cls_scores, bbox_preds = predictions
        batch_size = cls_scores.shape[0]
        device = cls_scores.device
        
        # Initialize total loss
        total_loss = torch.tensor(0., device=device)
        cls_loss = torch.tensor(0., device=device)
        box_loss = torch.tensor(0., device=device)
        
        for batch_idx in range(batch_size):
            # Get target boxes and labels for this batch item
            target_boxes = targets[batch_idx]['boxes']  # (num_boxes, 4)
            target_labels = targets[batch_idx]['labels']  # (num_boxes,)
            
            # Get predictions for this batch item
            batch_cls = cls_scores[batch_idx]  # (num_anchors, num_classes, H, W)
            batch_bbox = bbox_preds[batch_idx]  # (num_anchors, 4, H, W)
            
            # Get dimensions
            num_anchors, num_classes, H, W = batch_cls.shape
            
            # Create target mask
            target_mask = torch.zeros((H, W), dtype=torch.long, device=device)
            target_bbox_mask = torch.zeros((num_anchors, H, W, 4), dtype=torch.float32, device=device)
            valid_mask = torch.zeros((num_anchors, H, W), dtype=torch.bool, device=device)
            
            # Fill background by default
            target_mask.fill_(0)
            
            if len(target_boxes) > 0:
                for box, label in zip(target_boxes, target_labels):
                    x1, y1, x2, y2 = box.int()
                    # Ensure indices are within bounds
                    x1, x2 = torch.clamp(x1, 0, W-1), torch.clamp(x2, 0, W-1)
                    y1, y2 = torch.clamp(y1, 0, H-1), torch.clamp(y2, 0, H-1)
                    
                    # Fill the box area with the corresponding label
                    target_mask[y1:y2+1, x1:x2+1] = label
                    
                    # Fill bbox targets and valid mask for all anchors
                    for anchor_idx in range(num_anchors):
                        target_bbox_mask[anchor_idx, y1:y2+1, x1:x2+1] = box
                        valid_mask[anchor_idx, y1:y2+1, x1:x2+1] = True
            
            # Expand target_mask for all anchors
            target_mask = target_mask.unsqueeze(0).expand(num_anchors, -1, -1)  # (num_anchors, H, W)
            
            # Reshape predictions and targets for focal loss
            cls_preds = batch_cls.reshape(num_anchors, num_classes, -1)  # (num_anchors, num_classes, H*W)
            cls_preds = cls_preds.permute(0, 2, 1)  # (num_anchors, H*W, num_classes)
            cls_preds = cls_preds.reshape(-1, num_classes)  # (num_anchors*H*W, num_classes)
            
            cls_targets = target_mask.reshape(-1)  # (num_anchors*H*W)
            
            # Compute focal loss for classification
            probs = F.softmax(cls_preds, dim=1)
            target_probs = torch.gather(probs, 1, cls_targets.unsqueeze(1)).squeeze(1)
            focal_weights = (1 - target_probs) ** self.gamma
            alpha_weights = self.alpha[cls_targets]
            
            ce_loss = F.cross_entropy(cls_preds, cls_targets, reduction='none')
            batch_cls_loss = (focal_weights * alpha_weights * ce_loss).mean()
            
            # Compute bbox loss only for valid areas
            if valid_mask.any():
                # Reshape bbox predictions to match target format
                bbox_preds_reshape = batch_bbox.permute(0, 2, 3, 1)  # (num_anchors, H, W, 4)
                
                # Apply valid mask and compute loss
                bbox_preds_valid = bbox_preds_reshape[valid_mask]
                bbox_targets_valid = target_bbox_mask[valid_mask]
                batch_box_loss = self.bbox_loss(bbox_preds_valid, bbox_targets_valid)
            else:
                batch_box_loss = torch.tensor(0., device=device)
            
            # Combine losses
            cls_loss += batch_cls_loss
            box_loss += batch_box_loss
        
        # Average over batch
        cls_loss = cls_loss / batch_size
        box_loss = box_loss / batch_size
        total_loss = cls_loss + box_loss
        
        return total_loss, {'cls_loss': cls_loss.item(), 'box_loss': box_loss.item()}


# def create_focal_loss(num_classes=4, background_weight=0.1, gamma=2.0):
#     """
#     Factory function to create focal loss with default parameters
    
#     Args:
#         num_classes (int): Number of classes including background
#         background_weight (float): Weight for background class
#         gamma (float): Focal loss gamma parameter
#     """
#     # Create alpha weights with reduced background weight
#     alpha = torch.ones(num_classes)
#     alpha[0] = background_weight  # Set background weight
    
#     return DetectionLoss(alpha=alpha, gamma=gamma)


# Set random seeds for reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

    
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, print_every=5):
    """
    Train the model and print training/validation loss every k epochs
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        num_epochs: Number of epochs to train
        device: Device to train on (cuda/cpu)
        print_every: Print metrics every n epochs
    """
    
    # Move model to device
    model = model.to(device)
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        running_loss = 0.0
        batch_idx = -1
        
        for inputs, targets in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            model.train()
            # Move data to device
            if isinstance(inputs, (tuple, list)):
                inputs = torch.stack(inputs).to(device)
            else:
                inputs = image_tuple.to(device)
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v 
                       for k, v in t.items()} for t in targets]
#             inputs = inputs.to(device)
#             labels = labels.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             import pdb;pdb.set_trace()
            cls_scores, bbox_preds = model(inputs)
#             if len(targets) != cls_scores.shape[0]:
#                 import pdb;pdb.set_trace()
            try:
                loss, loss_dict = criterion((cls_scores, bbox_preds), targets)
            except:
                import pdb;pdb.set_trace()
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
            batch_idx += 1
            if batch_idx % 50 == 0:
                print(f'Loss: {loss.item():.4f}, '
                      f'Cls Loss: {loss_dict["cls_loss"]:.4f}, '
                      f'Box Loss: {loss_dict["box_loss"]:.4f}')
                
                model.eval()
                running_val_loss = 0.0

                with torch.no_grad():
                    for i, (inputs, targets) in enumerate(val_loader):
                        if i > 20: # only a subset
                            break
                        if isinstance(inputs, (tuple, list)):
                            inputs = torch.stack(inputs).to(device)
                        else:
                            inputs = image_tuple.to(device)
                        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v 
                               for k, v in t.items()} for t in targets]

                        outputs = model(inputs)
                        loss, _ = criterion(outputs, targets)
                        running_val_loss += loss.item() * inputs.size(0)

                epoch_val_loss = running_val_loss / (i+1)
                print(f'Validation Loss: {epoch_val_loss:.4f}')
#                 import pdb;pdb.set_trace()
        
        epoch_train_loss = running_loss / len(train_loader.dataset)
        
        # Validation phase
        model.eval()
        running_val_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                if isinstance(inputs, (tuple, list)):
                    inputs = torch.stack(inputs).to(device)
                else:
                    inputs = image_tuple.to(device)
                targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v 
                       for k, v in t.items()} for t in targets]
                
                outputs = model(inputs)
                try:
                    loss, _ = criterion(outputs, targets)
                except:
                    import pdb;pdb.set_trace()
                
                running_val_loss += loss.item() * inputs.size(0)
        
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        
        # Save best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            print("Best loss at epoch", epoch+1)
            torch.save(model.state_dict(), 'best_model.pth')
            
        
        # Print metrics every print_every epochs
        if (epoch + 1) % print_every == 1:
            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'Training Loss: {epoch_train_loss:.4f}')
            print(f'Validation Loss: {epoch_val_loss:.4f}')
            print('-' * 40)

            
def get_args():
    parser = ArgumentParser()
    parser.add_argument("--config_file", default="configs/imagenet/resnet101_timm.yaml")
#     parser.add_argument('--split', type=str, default="test")
    return parser.parse_args()


def main(model_name, ckpt_path, dataset_name, data_path, batch_size, fname):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_workers = 4
    batch_size = 32

    train_annotation_df = get_annotarion_dataframe(train_folder_list)
    
    clips = {}
    img_dir = Path(os.path.join(database_path, 'dayTrain/dayTrain'))
    print(img_dir)

    # Walk through all clip directories
    for clip_dir in img_dir.glob('dayClip*/frames'):
        # Extract clip name from path
        clip_name = clip_dir.parents[0].name + '--'  # e.g., 'dayClip13--'

        # Get all image files in frames directory
        frame_files = [f for f in clip_dir.glob('*.jpg')]

        # Extract frame numbers
        for frame_path in frame_files:
            # Extract frame number from filename (e.g., '00000' from 'dayClip13--00000.jpg')
            frame_num = int(frame_path.stem.split('--')[-1])

            if clip_name.strip('-') not in clips:
                clips[clip_name.strip('-')] = set()
            clips[clip_name.strip('-')].add(frame_num)

    positive_frames = set(train_annotation_df['filename'].unique())
    for clip_name, frame_nums in clips.items():
        for frame_num in frame_nums:
            filename = f"{clip_name}--{frame_num:05d}.jpg"
            if filename not in positive_frames:
                train_annotation_df = pd.concat([pd.DataFrame([[filename,'background', 0,0,0,0,f"{img_dir}/{clip_name}/frames/{filename}"]], 
                                                              columns=train_annotation_df.columns), 
                                                 train_annotation_df], 
                                                ignore_index=True)

    target_classes = train_annotation_df['target'].unique()
    target_classes.sort()

    def resample_dataset(annotation_df, n_samples):
        df_resample_list = list()
        for target in target_classes:
            df = annotation_df[annotation_df['target'] == target].copy()
            df_r = resample(df, n_samples=n_samples, random_state=42)
            df_resample_list.append(df_r)
        return pd.concat(df_resample_list).reset_index(drop=True)

    train_annotation_df = resample_dataset(train_annotation_df, n_samples_per_class)
    print(train_annotation_df['target'].value_counts())
    
    train_df, test_df = split_dataframe(train_annotation_df, random_state=42)
    train_df, val_df = split_dataframe(train_df, random_state=420)
    
    train_loader = create_traffic_light_data_loader(
        df=train_df,
        img_dir='/home/ttran02/.cache/kagglehub/datasets/mbornoe/lisa-traffic-light-dataset/versions/2/dayTrain/dayTrain',
        batch_size=batch_size,
        transform=get_transforms(train=True),
    )
    val_loader = create_traffic_light_data_loader(
        df=val_df,
        img_dir='/home/ttran02/.cache/kagglehub/datasets/mbornoe/lisa-traffic-light-dataset/versions/2/dayTrain/dayTrain',
        batch_size=batch_size,
        transform=get_transforms(train=False),
    )
#     train_dataset = get_dataset(dataset_name)(data_path=data_path,
#                                         preprocessing=False,
#                                         split="train", )
#     val_dataset = get_dataset(dataset_name)(data_path=data_path,
#                                         preprocessing=False,
#                                         split="val", )
#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=batch_size,
#         shuffle=True,
#         num_workers=num_workers,
#         pin_memory=True,
#         worker_init_fn=seed_worker,
#     )
#     val_loader = DataLoader(
#         val_dataset,
#         batch_size=batch_size,
#         shuffle=False,  # No need to shuffle validation data
#         num_workers=num_workers,
#         pin_memory=True,
#         worker_init_fn=seed_worker,
#     )

    model = get_fn_model_loader(model_name)(n_class=4, ckpt_path=ckpt_path).to(device)
    
    # Define loss function and optimizer
#     criterion = nn.CrossEntropyLoss()
    criterion = DetectionLoss(num_classes=4, background_weight=0.1, 
                              gamma=2.0).to(device)
#     criterion = create_focal_loss(
#         num_classes=4,  # background, stop, go, warning
#         background_weight=0.1,  # Reduce background influence
#         gamma=2.0  # Standard focal loss gamma
#     )
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training parameters
    num_epochs = 100
    print_every = 2
    
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=num_epochs,
        device=device,
        print_every=print_every
    )
    


if __name__ == "__main__":
    args = get_args()

    config = load_config(args.config_file)

    model_name = config['model_name']
    dataset_name = config['dataset_name']
    batch_size = config.get('batch_size', 32)

    data_path = config.get('data_path', None)
    ckpt_path = config.get('ckpt_path', None)

    fname = f"{model_name}_{dataset_name}"

    main(model_name, ckpt_path, dataset_name, data_path, batch_size, fname)
