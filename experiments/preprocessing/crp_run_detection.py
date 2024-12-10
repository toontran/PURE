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

class TrafficLightDataset(Dataset):
    def __init__(self, df, img_dir, target_size=(64,64), transform=None, 
                 scaling_factor=4):
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
            
#         image = np.array(image)
        
        # Apply any additional transforms
        if self.additional_transform:
            image = self.additional_transform(image)
        
        # Convert PIL to tensor if not done by transforms
        if isinstance(image, Image.Image):
            image = torchvision.transforms.functional.to_tensor(image)
        
        # Create target dictionary
#         target = {
#             'boxes': boxes,
#             'labels': labels,
#             'image_id': torch.tensor([idx]),
#             'area': (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
#             'iscrowd': torch.zeros((len(boxes),), dtype=torch.int64),
#             'filename': filename,
#         }
        target = labels[-1].item()
        
        return image, target


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

# Set random seeds for reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--config_file", default="configs/imagenet/resnet101_timm.yaml")
    parser.add_argument('--split', type=str, default="test")
    parser.add_argument('--ckpt_path', type=str, default=None)
    return parser.parse_args()


def main(model_name, ckpt_path, dataset_name, data_path, batch_size, fname, split):
    SPLIT = split

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

    if split == "train":
        df = train_df
        transforms_ = get_transforms(train=True)
    elif split == "val":
        df = val_df
        transforms_ = get_transforms(train=False)
    else:
        df = test_df
        transforms_ = get_transforms(train=False)
        
    dataset = TrafficLightDataset(
        df=df,
        img_dir='/home/ttran02/.cache/kagglehub/datasets/mbornoe/lisa-traffic-light-dataset/versions/2/dayTrain/dayTrain',
        transform=transforms_
    )

    model = get_fn_model_loader(model_name)(n_class=4, ckpt_path=ckpt_path).to(device)
    model.load_state_dict(torch.load('best_model.pth'))

    original_forward = model.forward
    
    def new_forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get original outputs using the stored forward method
        cls_scores, _ = original_forward(x)
        
        # cls_scores shape: (batch_size, num_anchors, num_classes + 1, H, W)
        batch_size, num_anchors, num_classes, H, W = cls_scores.shape
        
        # Apply softmax along the classes dimension to get probabilities
        cls_probs = torch.softmax(cls_scores, dim=2)
        
        # Sum the probabilities for traffic light classes (labels 1, 2, 3 for stop, go, warning)
        # Exclude background class (label 0)
        traffic_light_probs = cls_probs[:, :, 1:, :, :].sum(dim=2)  # (batch_size, num_anchors, H, W)
        
        # Find the anchor with highest traffic light probability for each spatial location
        best_anchor_scores, best_anchor_indices = traffic_light_probs.max(dim=1)  # (batch_size, H, W)
        
        return cls_scores[:, best_anchor_indices, :, :, :]
    
    # Bind the new forward method to the model
    model.forward = new_forward.__get__(model)
    model.eval()

    canonizers = get_canonizer(model_name)
    composite = EpsilonPlusFlat(canonizers)
    cc = ChannelConcept()
    layer_names = get_layer_names_model(model, model_name)
    print(layer_names)
    layer_map = {layer: cc for layer in layer_names}

    attribution = CondAttribution(model)

    os.makedirs(f"crp_files", exist_ok=True)
    
#     import pdb;pdb.set_trace()

    from torchvision import transforms
    fv = FeatureVisualization(attribution, dataset, layer_map, preprocess_fn=transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ]),
                              path=f"crp_files/{fname}_{SPLIT}", max_target="max", abs_norm=False)

    fv.ActMax.SAMPLE_SIZE = 100
    fv.run(composite, 0, len(dataset), batch_size=batch_size)


if __name__ == "__main__":
    args = get_args()

    config = load_config(args.config_file)

    model_name = config['model_name']
    dataset_name = config['dataset_name']
    batch_size = config.get('batch_size', 32)

    data_path = config.get('data_path', None)
    ckpt_path = config.get('ckpt_path', None)
    split = args.split

    fname = f"{model_name}_{dataset_name}"

    main(model_name, ckpt_path, dataset_name, data_path, batch_size, fname, split)
