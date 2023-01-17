from torch.utils.data import Dataset
from torchvision import transforms as T
import os
from PIL import Image
import pandas as pd

class ImageLoader(Dataset):
    def __init__(self, path, transforms=None):
        self.imgs = []
        for img in os.listdir(os.path.join(path, 'images')):
            _, hair_label, eye_label = img.split('.')[0].split('_')
            self.imgs.append((os.path.join(path, 'images', img), int(hair_label), int(eye_label)))

        if transforms is None:
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            self.transforms = T.Compose([
                T.ToTensor(),
                normalize
            ])
        else:
            self.transforms = transforms
        
    def __getitem__(self, item):
        img_path = self.imgs[item][0]
        hair_label = self.imgs[item][1]
        eye_label = self.imgs[item][2]

        data = Image.open(img_path)
        if data.mode != 'RGB':
            data = data.convert('RGB')
        data = self.transforms(data)

        return data, hair_label, eye_label

    def __len__(self):
        return len(self.imgs)