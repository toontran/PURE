import timm
import torch
import torch.hub
from timm.models import checkpoint_seq
from utils.helper import InspectionLayer
from utils.lrp_canonizers import ResNetCanonizer
import torch.nn as nn


class DetectionHead(nn.Module):
    def __init__(self, in_channels, num_classes, num_anchors=3):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
#         # Common features
        self.conv_shared = nn.Sequential(
            nn.Conv2d(in_channels, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        # Classification head
        # For each anchor: num_classes 
        self.conv_cls = nn.Conv2d(512, num_anchors * (num_classes), kernel_size=1)
#         self.conv_cls = nn.Conv2d(in_channels, num_anchors * (num_classes), kernel_size=1)
        
        # Bounding box regression head
        # For each anchor: 4 values (x, y, w, h)
        self.conv_bbox = nn.Conv2d(512, num_anchors * 4, kernel_size=1)
#         self.conv_bbox = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1)

    def forward(self, x):
        x = self.conv_shared(x)
        
        # Classification branch
        cls_scores = self.conv_cls(x)
        batch_size = cls_scores.shape[0]
        cls_scores = cls_scores.view(batch_size, self.num_anchors, self.num_classes, 
                                   cls_scores.shape[-2], cls_scores.shape[-1])
        
        # Bounding box regression branch
        bbox_preds = self.conv_bbox(x)
        bbox_preds = bbox_preds.view(batch_size, self.num_anchors, 4,
                                   bbox_preds.shape[-2], bbox_preds.shape[-1])
        
        return cls_scores, bbox_preds


def get_resnet18(ckpt_path=None, pretrained=True, n_class: int = None) -> torch.nn.Module:
    return get_resnet_timm('resnet18.a1_in1k', ckpt_path, pretrained, n_class)


def get_resnet34_timm(ckpt_path=None, pretrained=True, n_class=None) -> torch.nn.Module:
    return get_resnet_timm('resnet34.a1_in1k', ckpt_path, pretrained, n_class)


def get_resnet50_timm(ckpt_path=None, pretrained=True, n_class: int = None) -> torch.nn.Module:
    return get_resnet_timm('resnet50.a1_in1k', ckpt_path, pretrained, n_class)

def get_resnet101_timm(ckpt_path=None, pretrained=True, n_class: int = None) -> torch.nn.Module:
    return get_resnet_timm('resnet101.a1_in1k', ckpt_path, pretrained, n_class)

# def get_resnet_timm(name, ckpt_path=None, pretrained=True, n_class: int = None) -> torch.nn.Module:
    
#     model = timm.create_model(name, pretrained, num_classes=n_class) #global_pool=None 11221
#     # print(timm.data.resolve_model_data_config(model))
#     #model = m(weights=weights)

#     if n_class and n_class != 1000:
#         num_in = model.fc.in_features
#         model.fc = torch.nn.Linear(num_in, n_class, bias=True)
#     if ckpt_path:
#         checkpoint = torch.load(ckpt_path)
#         if "state_dict" in checkpoint:
#             checkpoint = checkpoint["state_dict"]
#         if "module" in list(checkpoint.keys())[0]:
#             checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}
#         model.load_state_dict(checkpoint)

#     model.block_0 = InspectionLayer()
#     model.block_1 = InspectionLayer()
#     model.block_2 = InspectionLayer()
#     model.block_3 = InspectionLayer()

#     model.forward_features = forward_features_.__get__(model)
#     return model

def get_resnet_timm(name, ckpt_path=None, pretrained=True, n_class: int = None) -> torch.nn.Module:
    model = timm.create_model(name, pretrained, num_classes=n_class, global_pool='')
    
    if ckpt_path:
        checkpoint = torch.load(ckpt_path)
        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        if "module" in list(checkpoint.keys())[0]:
            checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}
        model.load_state_dict(checkpoint, strict=False)  # Changed to strict=False
    
    # Add inspection layers
    model.block_0 = InspectionLayer()
    model.block_1 = InspectionLayer()
    model.block_2 = InspectionLayer()
    model.block_3 = InspectionLayer()
#     model.block_4 = InspectionLayer()
    
    # Get the number of channels from the last layer
    if hasattr(model, 'layer4'):
        last_channel = model.layer4[-1].conv3.out_channels if hasattr(model.layer4[-1], 'conv3') else model.layer4[-1].conv2.out_channels
    else:
        last_channel = model.feature_info[-1]['num_chs']
    
    print("before detection head:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    
    # Replace classification head with detection head
    model.detection_head = DetectionHead(in_channels=last_channel, num_classes=n_class)
    
    # Remove the original classification head
    if hasattr(model, 'fc'):
        delattr(model, 'fc')
    if hasattr(model, 'global_pool'):
        delattr(model, 'global_pool')
    
    # Bind new forward methods
    model.forward_features = forward_features_.__get__(model)
    model.forward = forward_detection.__get__(model)
    print("after detection head:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    
    return model

def forward_features_(self, x: torch.Tensor) -> torch.Tensor:
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.act1(x)
    x = self.maxpool(x)

    if self.grad_checkpointing and not torch.jit.is_scripting():
        x = checkpoint_seq([self.layer1, self.layer2, self.layer3, self.layer4], x, flatten=True)
    else:
        x = self.layer1(x)
        x = self.block_0(x)  # added identity
        x = self.layer2(x)
        x = self.block_1(x)  # added identity
        x = self.layer3(x)
        x = self.block_2(x)  # added identity
        x = self.layer4(x)
        x = self.block_3(x)  # added identity
    return x

# def forward_(self, x: torch.Tensor) -> torch.Tensor:
#     x = self.forward_features(x)
#     x = self.forward_head(x)
#     x = self.selection(x)
#     return x

def forward_detection(self, x: torch.Tensor):
    features = self.forward_features(x)
    cls_scores, bbox_preds = self.detection_head(features)
    return cls_scores, bbox_preds


def get_resnet_canonizer():
    return ResNetCanonizer()




