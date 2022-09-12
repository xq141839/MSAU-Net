import os
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset


#Dataset Loader
class medical_img_data(Dataset):
        def __init__(self,path, transform=None):
            self.path = path
            self.folders = os.listdir(path)
            self.transforms = transform
        
        def __len__(self):
            return len(self.folders)
              
        
        def __getitem__(self,idx):
            image_path= os.path.join(self.path,self.folders[idx],'images/',self.folders[idx])
            mask_path = os.path.join(self.path,self.folders[idx],'masks/',self.folders[idx])
            
            img = io.imread(f'{image_path}.png')[:,:,:3].astype('float32')
            
            mask = io.imread(f'{mask_path}.png', as_gray=True)

            augmented = self.transforms(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']

            return (img, mask) 
