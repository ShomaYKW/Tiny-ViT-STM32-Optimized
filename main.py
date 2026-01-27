import torch

from train.train import run_training
from dataloader.CIFAR10 import get_loaders
from model.model import ViT


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

if __name__ =="__main__":

    train_loader, test_loader = get_loaders(batch_size=128)

    device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    print(f"using: {device}")

    model = ViT(**custom_config_CIFAR10)
    model.to(device)

    trained_model = run_training(
        train_loader = train_loader, test_loader = test_loader, model = model, num_epochs = 50, lr =  3e-4
    )

    torch.save(trained_model.state_dict(), "pathsaver/vit_cifar10.pth")
    print("Model saved to vit_cifar10.pth")


    