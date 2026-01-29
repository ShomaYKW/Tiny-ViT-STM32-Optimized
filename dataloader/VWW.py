import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class VWWDataset(Dataset):
    
    def __init__(self, root_dir, split='train', transform=None):
        
        self.transform = transform
        self.image_paths = []
        self.labels = []  #

        # Define the specific split directory 
        split_dir = os.path.join(root_dir, split)
        
        if not os.path.exists(split_dir):
            raise FileNotFoundError(f"Split folder not found: {split_dir}")

        valid_classes = sorted(os.listdir(split_dir))
        
        print(f"Found classes for {split}: {valid_classes}")

        for class_name in valid_classes:
            class_path = os.path.join(split_dir, class_name)
            if not os.path.isdir(class_path):
                continue

            
            if class_name.lower() in ['person', '1', 'positive']:
                label = 1
            else:
                label = 0 # background

            # Collect images
            for img_name in os.listdir(class_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    self.image_paths.append(os.path.join(class_path, img_name))
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # Open image and convert to RGB
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a dummy image or handle error appropriately in real training
            return self.__getitem__((idx + 1) % len(self)) 

        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

def get_vww_loaders(root='./data/visual_wake_words', batch_size=32, img_size=96):
    """
    Create Train and Test loaders for VWW.
    """
    
    # Define Transforms
    transform_train = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_test = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load Datasets
    train_dataset = VWWDataset(root_dir=root, split='train', transform=transform_train)
    
    if os.path.exists(os.path.join(root, 'val')):
        test_split_name = 'val'
    else:
        test_split_name = 'test'
        
    test_dataset = VWWDataset(root_dir=root, split=test_split_name, transform=transform_test)
    
    # Check if data was found
    if len(train_dataset) == 0:
        raise RuntimeError(f"No images found in '{root}/train'. Check paths.")

    print(f"Loaded VWW: {len(train_dataset)} Training samples, {len(test_dataset)} Test samples.")

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, test_loader