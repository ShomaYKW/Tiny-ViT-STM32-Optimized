import torch

from train.train import run_training
from dataloader.CIFAR10 import get_CIFAR10_loaders
from dataloader.VWW import get_mvtec_loaders
from model.model import ViT
from debug.debug import debug_single_batch

custom_config_CIFAR10 = {
    "img_size" : 32,
    "in_chans" : 3,
    "patch_size" : 4,
    "embed_dim": 384,
    "depth": 6,
    "n_heads":6,
    "qkv_bias": True,
    "mlp_ratio": 4
    }

costom_config_MVTec = {
    "img_size": 224,     
    "in_chans": 3,
    "patch_size": 16,
    "n_classes": 2,     
    "embed_dim": 384,
    "depth": 6,
    "n_heads": 6,
    "qkv_bias": True,
    "mlp_ratio": 4
}


if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    print(f"using: {device}")

    train_loader_CIFAR10, test_loader_CIFAR10 = get_CIFAR10_loaders(batch_size=128)

    """train_loader_VWW, test_loader_VWW = get_mvtec_loaders(
        root='./data/mvtec_anomaly_detection', 
        category='bottle', 
        batch_size=32
    )
    """
    model = ViT(**custom_config_CIFAR10)
    model.to(device)
    
    debug_single_batch(model, train_loader_CIFAR10, device)