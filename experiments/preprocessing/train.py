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
from utils.helper import load_config, get_layer_names_model
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


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
        model.train()
        running_loss = 0.0
        
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            import pdb;pdb.set_trace()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
        
        epoch_train_loss = running_loss / len(train_loader.dataset)
        
        # Validation phase
        model.eval()
        running_val_loss = 0.0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                running_val_loss += loss.item() * inputs.size(0)
        
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        
        # Save best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
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

    train_dataset = get_dataset(dataset_name)(data_path=data_path,
                                        preprocessing=False,
                                        split="train", )
    val_dataset = get_dataset(dataset_name)(data_path=data_path,
                                        preprocessing=False,
                                        split="val", )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # No need to shuffle validation data
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
    )

    model = get_fn_model_loader(model_name)(n_class=train_dataset.num_classes, ckpt_path=ckpt_path).to(device)
    print("Num params:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training parameters
    num_epochs = 30
    print_every = 5
    
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
