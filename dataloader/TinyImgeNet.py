import os
import requests
import zipfile
import shutil
import torch 
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

def get_TinyImageNet_loaders(batch_size = 128, num_classes = 10):
    
    data_dir = './tiny-imagenet-200'
    url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'

    if not os.path.exists(data_dir):
        print("downloading Tiny ImageNet")
        r = requests.get(url, stream=True)
        with open('tiny-imagenet-200.zip', 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk: f.write(chunk)

    print("now unzipping")
    with zipfile.ZipFile('tiny-imagenet-200.zip' , 'r') as zip_ref:
        zip_ref.extractall('.')

    print("Reformatting validation set")
    val_dir = os.path.join(data_dir, 'val')
    img_dir = os.path.join(val_dir, 'images')
    annot_file = os.path.join(val_dir, 'val_annotations.txt')
        
    with open(annot_file, 'r') as f:
        for line in f:
            parts = line.split('\t')
            filename, class_id = parts[0], parts[1]
                
                
            class_folder = os.path.join(val_dir, class_id)
            os.makedirs(class_folder, exist_ok=True)
                
                
            src = os.path.join(img_dir, filename)
            dst = os.path.join(class_folder, filename)
            if os.path.exists(src):
                shutil.move(src, dst)
        
        
        if os.path.exists(img_dir):
            os.rmdir(img_dir)

    
    stats = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(*stats)
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*stats)
    ])

    # load data
    train_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'train'), transform=train_transform)
    test_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'val'), transform=test_transform)

    class_indices = list(range(len(train_dataset.classes)))[:num_classes]
    target_classes = [train_dataset.classes[i] for i in class_indices]
    
    # Filter Train
    train_idx = [i for i, label in enumerate(train_dataset.targets) if label in class_indices]
    train_subset = Subset(train_dataset, train_idx)
    
    
    test_idx = [i for i, label in enumerate(test_dataset.targets) if label in class_indices]
    test_subset = Subset(test_dataset, test_idx)

    print(f"Data Loaded! Using {num_classes} classes.")
    print(f"Train size: {len(train_subset)} | Test size: {len(test_subset)}")

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, test_loader