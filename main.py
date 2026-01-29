import torch

from model.model import ViT
from model.pretrained_model import get_vit_tiny_trained

from dataloader.CIFAR10 import get_CIFAR10_loaders
from dataloader.MVTec import get_mvtec_loaders
from dataloader.VWW import get_vww_loaders
from dataloader.TinyImgeNet import get_TinyImageNet_loaders

from train.train import run_training, run_training_MVTec
from train.inference import run_inference


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



if __name__ =="__main__":

    device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    print(f"using: {device}")

    #get the dataset
    train_loader_CIFAR10, test_loader_CIFAR10 = get_CIFAR10_loaders(batch_size=128)
    train_loader_MVtec_bottle, test_loader_MVTec_bottle = get_mvtec_loaders(
        root='/home/sxy901/ViT_P2/data/mvtec_anomaly_detection',
        category='bottle',  
        batch_size=32, 
        img_size=224, 
        split_ratio=0.8
        )
    #train_loader_VWW, test_loader_VWW = get_vww_loaders()

    train_loader_TinyImageNet, test_loader_TinyImageNet = get_TinyImageNet_loaders(batch_size=128, num_classes=10)

    #get the base model 
    pretrained_tiny_model = get_vit_tiny_trained()
    pretrained_tiny_model.to(device)
    base_model_CIFAR10 = ViT(**custom_config_CIFAR10)
    base_model_CIFAR10.to(device)
    base_model_MVTec = ViT(**custom_config_MVTec)
    base_model_MVTec.to(device)
    base_model_VWW = ViT(**custom_config_VWW)
    base_model_VWW.to(device)
    base_model_TinyImageNet = ViT(**custom_config_TinyImageNet)
    base_model_TinyImageNet.to(device)

    """
    print("now training on CIFAR10")
    trained_model_CIFAR10 = run_training(
        train_loader = train_loader_CIFAR10, test_loader = test_loader_CIFAR10, model = base_model_CIFAR10, num_epochs = 200, lr =  3e-4, weight_decay= 0.05
    )
    torch.save(trained_model_CIFAR10.state_dict(), "pathsaver/vit_cifar10.pth")
    print("Models saved")
    """

    """
    print("now training on MVTec")
    trained_model_MVTec = run_training_MVTec(
        train_loader = train_loader_MVtec_bottle, test_loader = test_loader_MVTec_bottle, model = base_model_MVTec, num_epochs = 200, lr =  1e-4, weight_decay= 1e-4
    )
    torch.save(trained_model_MVTec.state_dict(), "pathsaver/vit_MVTec.pth")
    print("Models saved")
    """
    
    print("now training on TinyImagwNet")
    trained_model_TinyImageNet = run_training(
        train_loader = train_loader_TinyImageNet, test_loader = test_loader_TinyImageNet, model = base_model_TinyImageNet, num_epochs = 200, lr =  1e-4, weight_decay= 1e-4
    )
    torch.save(trained_model_TinyImageNet.state_dict(), "pathsaver/vit_tinyimagenet.pth")
    print("Models saved")




    