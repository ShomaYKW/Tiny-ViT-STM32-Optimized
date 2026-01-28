import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

class MVTecSupervisedDataset(Dataset):
    """
    combine train and test dataset for better training for the model 
    """

    def __init__(self, root_dir, category, transform = None):

        #transform to be applied on sample
        self.transform = transform
        self.image_paths = []
        self.labels = []

        #define category path
        category_dir = os.path.join(root_dir, category)

        #check for non-existant 
        if not os.path.exists(category_dir):
            raise FileNotFoundError
        
        #collect both train and test dataset for a supervised split
        for split in ['train', 'test']:
            split_path = os.path.join(category_dir, split)

            if not os.path.exists:
                continue

            #loop through the sub folders
            for subfolder in os.listdir(split_path):
                subfolder_path = os.path.join(split_path, subfolder)
                if not os.path.isdir(subfolder_path):
                    continue

                #labeling
                # if the folder name is good , label 1, else 0
                label = 0 if subfolder == 'good' else 0

                for img_name in os.listdir(subfolder_path):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                        self.image_paths.append(os.path.join(subfolder_path, img_name))
                        self.labels.append(label)

#get the length 
    def __len__(self):
        return len(self.image_paths)

#get the image and convert to RGB
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        # Open image and convert to RGB 
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

def get_mvtec_loaders(root='./data/mvtec_anomaly_detection', category='bottle', batch_size=32, img_size=224, split_ratio=0.8):
    """
    Create Train and Test loaders for a specific MVTec category.
    """
    
    # 1. Define Transforms
    # MVTec requires high resolution (224) to see small defects.
    # use ImageNet normalization stats as standard practice.
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 2. Load the full dataset (Train + Test combined)
    full_dataset = MVTecSupervisedDataset(root_dir=root, category=category, transform=transform)
    
    # Check if data was found
    if len(full_dataset) == 0:
        raise RuntimeError(f"No images found for category '{category}' in '{root}'. Check paths.")

    # 3. Randomly Split into Train and Validation
    train_size = int(split_ratio * len(full_dataset))
    test_size = len(full_dataset) - train_size
    
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    print(f"Loaded MVTec '{category}': {train_size} Training samples, {test_size} Test samples.")

    # 4. Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, test_loader
                      

