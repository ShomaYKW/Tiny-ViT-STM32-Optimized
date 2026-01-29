import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

class MVTecSupervisedDataset(Dataset):
    
    def __init__(self, root_dir, category, transform=None):
        """
        Args:
            root_dir (str): Path to the main MVTec folder.
            category (str): The object category to load (e.g., 'bottle', 'cable').
            transform (callable, optional): Transform to be applied on a sample.
        """
        self.transform = transform
        self.image_paths = []
        self.labels = []  

        # Define the category path 
        category_dir = os.path.join(root_dir, category)
        
        if not os.path.exists(category_dir):
            raise FileNotFoundError(f"Category folder not found: {category_dir}")

        # Loop through both 'train' and 'test' folders t
        for split in ['train', 'test']:
            split_path = os.path.join(category_dir, split)
            
            if not os.path.exists(split_path):
                continue

            # Loop through the subfolders 
            for subfolder in os.listdir(split_path):
                subfolder_path = os.path.join(split_path, subfolder)
                
                if not os.path.isdir(subfolder_path):
                    continue

                label = 0 if subfolder == 'good' else 1

                for img_name in os.listdir(subfolder_path):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                        self.image_paths.append(os.path.join(subfolder_path, img_name))
                        self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # Open image and convert to RGB 
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return self.__getitem__((idx + 1) % len(self))

        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

def get_mvtec_loaders(root, category, batch_size=32, img_size=224, split_ratio=0.8):
   
    
    #Define Transforms
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    #Load 
    full_dataset = MVTecSupervisedDataset(root_dir=root, category=category, transform=transform)
    
    #Check if data was found
    if len(full_dataset) == 0:
        raise RuntimeError(f"No images found. Check paths.")

    #Randomly Split into Train and Test
    train_size = int(split_ratio * len(full_dataset))
    test_size = len(full_dataset) - train_size
    
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    print(f"Loaded MVTec '{category}': {train_size} Training samples, {test_size} Test samples.")

    #Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, test_loader