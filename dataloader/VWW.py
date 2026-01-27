import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_loaders(batch_size=128, root='./data', num_workers=2):


    transform_224 = transforms.Compose([
        transforms.Resize((224, 224)),  
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
    ])

    # 2. Download and Load Training Data
    train_dataset = datasets.CIFAR10(
        root=root, 
        train=True, 
        download=True, 
        transform=transform_224
    )

    # 3. Download and Load Test Data
    test_dataset = datasets.CIFAR10(
        root=root, 
        train=False, 
        download=True, 
        transform=transform_224
    )

    # 4. Create DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, test_loader


