import os
import nibabel as nib
from monai.transforms import ScaleIntensity, NormalizeIntensity, Compose
import torch

# Definir el pipeline de transformaciones
transform = Compose([ScaleIntensity(minv=0.0, maxv=1.0),NormalizeIntensity(nonzero=True)])

def search_paths(base_path, pattern, all_data=True):
    rutas = []
    for root, __, files in os.walk(base_path):
        for file in files:
            if not file.endswith(pattern):
                continue
            if (not all_data) and ("ses-1" in file):
                continue
            rutas.append(os.path.join(root, file))
    return rutas


def normalice_and_save (output_path):

    preprocess_path = output_path + "derivatives/preprocessed"

    files = search_paths(preprocess_path, "brain_registered.nii.gz")

    transform = Compose([ScaleIntensity(minv=0.0, maxv=1.0),NormalizeIntensity(nonzero=True)])

    for file in files:
        img_o = nib.load(file).get_fdata()
        img_o_tensor = torch.tensor(img_o)
        imagen_nor = transform(img_o_tensor)
        ruta_norm = file.replace('brain_registered.nii.gz', 'brain_normalized.nii.gz')
        nib.save(nib.Nifti1Image(imagen_nor.numpy(), None), ruta_norm)
    
    return
