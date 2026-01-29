import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.models as models

def get_vit_tiny_trained():
    model = timm.create_model("vit_tiny_patch16_224", pretrained=True)
    
    return model