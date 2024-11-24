import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader
from utils.normalización import search_paths
import torch.nn.functional as F



class BrainDataset(torch.utils.data.Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        image_path = self.file_paths[idx]
        label = self.labels[idx]
        image = nib.load(image_path).get_fdata()
        image = image.astype(np.float32)
        #Probar
        image = np.expand_dims(image, axis=0)
        image = torch.tensor(image)
        #image = F.pad(image, (54, 54, 46, 47, 54, 54))
        #image = F.interpolate(image, size=(200, 91), mode='bilinear', align_corners=False)
        if self.transform:
            image = self.transform(image)
        return image, label
    

def load_data(output_path: str, batch_size: int, train: int = 0.7, transform=None):

    files = search_paths(output_path, "brain_normalized.nii.gz")
    labels =[0 if "RC41" in file else 1 for file in files] #Se toman los labels como dice la documentación de estos
    dataset = BrainDataset(files, labels, transform)
    train_size = int(train * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader

