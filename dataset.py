import torch.utils.data as data
from PIL import Image
import os
import torch
import random
import csv

class Dataset(data.Dataset):
    
    def __init__(self, img_dir, transform, csv_path=None):

        self.img_dir = img_dir
        self.csv_path = csv_path
        self.transform = transform
        self.data = []
        self.label = []
        self.fname = []
              
        # read data & labels
        if csv_path is not None:
            with open(csv_path) as csvfile:
                _csv = csv.reader(csvfile)
                next(_csv)
                for row in _csv:
                    img_path = os.path.join(img_dir, row[0])
                    image = Image.open(img_path).convert('RGB')
                    self.data.append(image)
                    self.fname.append(row[0])
                    self.label.append(int(row[1]))
        else:
            for x in sorted(os.listdir(img_dir)):
                img_path = os.path.join(img_dir, x)
                img = Image.open(img_path).convert('RGB')
                self.fname.append(x)
                self.data.append(img)
            
    def __getitem__(self, index):
        img = self.transform(self.data[index])
        fname = self.fname[index]
        if self.csv_path is not None:
            label = self.label[index]
            return img, fname, label
        else:
            return img, fname

    def __len__(self):
        return len(self.data)

"""
import pdb
from torchvision import transforms
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
dataset = Dataset('digit_data/digits/usps/train', 'digit_data/digits/usps/train.csv', transform)
pdb.set_trace()
"""