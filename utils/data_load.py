import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader
from utils.normalización import search_paths
import torch.nn.functional as F
from torchvision import transforms

class BalancedBrainDataset(torch.utils.data.Dataset):
    def __init__(self, original_dataset):
        self.original_dataset = original_dataset
        self.balanced_data = []
        for i in range(len(original_dataset)):
            data = original_dataset[i]
            if isinstance(data, list):
                self.balanced_data.extend(data)
            else:
                self.balanced_data.append(data)

    def __len__(self):
        return len(self.balanced_data)

    def __getitem__(self, idx):
        return self.balanced_data[idx]


# class BrainDataset(torch.utils.data.Dataset):
#     def __init__(self, file_paths, labels, transform=None):
#         self.file_paths = file_paths
#         self.labels = labels
#         self.transform = transform

#     def __len__(self):
#         return len(self.file_paths)

#     def __getitem__(self, idx):
#         image_path = self.file_paths[idx]
#         label = self.labels[idx]
#         image = nib.load(image_path).get_fdata()
#         image = image.astype(np.float32)
#         #Probar
#         image = np.expand_dims(image, axis=0)
#         image = torch.tensor(image)
#         image = F.pad(image, (54, 54, 46, 47, 54, 54))
#         #image = F.interpolate(image, size=(200, 91), mode='bilinear', align_corners=False)
#         if self.transform:
#             image = self.transform(image)
#         return image, label

class BrainDataset(torch.utils.data.Dataset):
    def __init__(self, file_paths, labels, transform=None, augmentation_transform=None, augmentation=False):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform
        self.augmentation_transform = augmentation_transform
        self.augmentation = augmentation

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        image_path = self.file_paths[idx]
        label = self.labels[idx]
        image = nib.load(image_path).get_fdata()
        image = image.astype(np.float32)
        image = np.expand_dims(image, axis=0)
        image = torch.tensor(image)
        #image = F.pad(image, (54, 54, 46, 47, 54, 54))
        if label == 0 and self.augmentation_transform and self.augmentation:
            augmented_image = self.augmentation_transform(image)
            return [(image, label), (augmented_image, label)]
        if self.transform:
            image = self.transform(image)
        return image, label

def load_data(output_path: str, batch_size: int, train: int = 0.7, transform=None, augmentation=True, all_data=False):

    files = search_paths(output_path, "brain_normalized.nii.gz", all_data)
    labels =[0 if "RC41" in file else 1 for file in files] #Se toman los labels como dice la documentación de estos

    augmentation_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(0.8),
        transforms.RandomVerticalFlip(0.8),
        transforms.RandomRotation(15)
    ])

    brain_dataset = BrainDataset(files, labels, augmentation_transform=augmentation_transform)
    #Duplicamos las imagenes... No es lo mejor pero xd
    if not augmentation:
        brain_dataset = BalancedBrainDataset(brain_dataset)
    
    train_size = int(train * len(brain_dataset))
    test_size = len(brain_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(brain_dataset, [train_size, test_size], generator=torch.Generator().manual_seed(23))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    dataset_info(train_dataset, "entrenamiento")
    dataset_info(test_dataset, "prueba")

    return train_loader, test_loader

def dataset_info(dataset, msg):
    contador = {0: 0, 1: 0}
    for _, label in dataset:
        contador[label] += 1
    print(f"Conteo de etiquetas en {msg}: {contador}")