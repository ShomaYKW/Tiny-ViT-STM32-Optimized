import torch
import os
import copy
from model.model import ViT
from dataloader.CIFAR10 import get_CIFAR10_loaders
from dataloader.MVTec import get_mvtec_loaders
from dataloader.VWW import get_vww_loaders
from dataloader.TinyImgeNet import get_TinyImageNet_loaders
from train.train import run_training, run_training_MVTec
from train.inference import run_inference


def CIFAR10_flow(device):

    print(f"using device for CIFAR10 process: {device}")

    custom_config_CIFAR10 = {
    "img_size" : 32,
    "in_chans" : 3,
    "patch_size" : 4,
    "embed_dim": 64,
    "depth": 6,
    "n_heads":4,
    "qkv_bias": True,
    "mlp_ratio": 2
    }

    CIFAR10_Trained_Path = "pathsaver/vit_cifar10.pth"
    train_loader_CIFAR10, test_loader_CIFAR10 = get_CIFAR10_loaders(batch_size=128)

    base_model_CIFAR10 = ViT(**custom_config_CIFAR10)
    base_model_CIFAR10.to(device)

    """
    print("now training on CIFAR10")
    trained_model_CIFAR10 = run_training(
        train_loader = train_loader_CIFAR10, test_loader = test_loader_CIFAR10, model = base_model_CIFAR10, num_epochs = 200, lr =  3e-4, weight_decay= 0.05
    )
    torch.save(trained_model_CIFAR10.state_dict(), "pathsaver/vit_cifar10.pth")
    print("Models saved")
    """

    print("loading ViT trained on CIFAR10")
    state_dict = torch.load(CIFAR10_Trained_Path, map_location=device)
    base_model_CIFAR10.load_state_dict(state_dict)
    print("Base model loaded, now running test inference")
    base_model_CIFAR10.eval()
    run_inference(test_loader = test_loader_CIFAR10, model = base_model_CIFAR10, num_epoch = 1, device = device)

    ("now pruning the model")
    





def MVTec_flow(device):

    print(f"using: {device}")

    custom_config_MVTec = {
    "img_size": 224,      
    "patch_size": 16,     
    "n_classes": 2,     
    "embed_dim": 64,     
    "depth": 6,          
    "n_heads": 4,
    "qkv_bias": True,
    "mlp_ratio": 2        
    }

    MVTec_Trained_Path = "pathsaver/vit_MVTec.pth"  
    train_loader_MVTec, test_loader_MVTec = get_mvtec_loaders(
        root='./data/mvtec_anomaly_detection', 
        category='bottle', 
        batch_size=32
    )


    print("now loading MVTec")
    base_model_MVTec = ViT(**custom_config_MVTec)
    """
    print("now training on MVTec")
    trained_model_MVTec = run_training_MVTec(
        train_loader = train_loader_MVtec_bottle, test_loader = test_loader_MVTec_bottle, model = base_model_MVTec, num_epochs = 200, lr =  1e-4, weight_decay= 1e-4
    )
    torch.save(trained_model_MVTec.state_dict(), "pathsaver/vit_MVTec.pth")
    print("Models saved")
    """
    print("loading ViT trained on MVTec")
    state_dict = torch.load(MVTec_Trained_Path, map_location=device)
    base_model_MVTec.load_state_dict(state_dict)
    print("Base model loaded, now running test inference")
    run_inference(test_loader = test_loader_MVTec, model = base_model_MVTec, num_epoch = 10, device = device)




    

def VWW_flow():

    print(f"using: {device}")

    custom_config_VWW = {
    "img_size": 96,       
    "in_chans": 3,
    "patch_size": 12,    
    "n_classes": 2,
    "embed_dim": 48,      
    "depth": 4,           
    "n_heads": 3,         
    "qkv_bias": False,    
    "mlp_ratio": 2
    }


def TinyImageNet_flow():
    custom_config_TinyImageNet = {
    "img_size": 64,       
    "in_chans": 3,
    "patch_size": 8,
    "n_classes": 10,    
    "embed_dim": 64,    
    "depth": 8,           
    "n_heads": 4,        
    "qkv_bias": True,
    "mlp_ratio": 2  
    }      
    
    TinyImageNet_Trained_Path = "pathsaver/vit_tinyimagenet.pth"
    train_loader_TinyImageNet, test_loader_TinyImageNet = get_TinyImageNet_loaders(batch_size=128, num_classes=10)

    base_model_TinyImageNet = ViT(**custom_config_TinyImageNet)
    base_model_TinyImageNet.to(device)

    print("now training on TinyImagwNet")
    trained_model_TinyImageNet = run_training(
        train_loader = train_loader_TinyImageNet, test_loader = test_loader_TinyImageNet, model = base_model_TinyImageNet, num_epochs = 200, lr =  1e-4, weight_decay= 1e-4
    )
    torch.save(trained_model_TinyImageNet.state_dict(), "pathsaver/vit_tinyimagenet.pth")
    print("Models saved")

    print("loading ViT trained on TinyImageNet")
    state_dict = torch.load(TinyImageNet_Trained_Path, map_location=device)
    base_model_TinyImageNet.load_state_dict(state_dict)
    print("Base model loaded, now running test inference")
    base_model_TinyImageNet.eval()
    run_inference(test_loader = test_loader_TinyImageNet, model = base_model_TinyImageNet, num_epoch = 1, device = device)

    ("now pruning the model")


    

if __name__ =="__main__":

    device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    print(f"using: {device}")

    CIFAR10_flow(device)

    
    