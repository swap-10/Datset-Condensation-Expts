import os
import os.path
import torch.utils.data as data
import torchvision.transforms as transforms
import torch
import numpy as np
import csv
import torchvision
from PIL import Image

categories = ['benign', 'malignant']

category_to_idx = {c: i  for i,c in enumerate(categories)}

class ISIC2016(data.Dataset):
    label2encode = {"benign": 0, "malignant": 1}
    
    def __init__(self, root, phase, transform=None):
        self.root = root
        self.train = phase == "train"
        self.transform = transform
        if self.train:
            self.images_path = os.path.join(self.root, "ISIC2016_TrainingImages")
            self.labels_path = os.path.join(self.root, "ISIC2016_TrainingLabels.csv")
        else:
            self.images_path = os.path.join(self.root, "ISIC2016_TestImages")
            self.labels_path = os.path.join(self.root, "ISIC2016_TestLabels.csv")
        self.labels = []
        self.targets = []

        with open(self.labels_path, 'r') as csvfile:
          print("print", self.labels_path)
          reader = csv.reader(csvfile)
          for row in reader:
              if self.train:
                  self.labels.append({"image_name": row[0], "label": ISIC2016.label2encode[row[1]]})
                  self.targets.append(ISIC2016.label2encode[row[1]])
              else:
                  self.labels.append({"image_name": row[0], "label": int(float(row[1]))})
                  self.targets.append(int(float(row[1])))
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_name = self.labels[idx]['image_name']
        label = self.labels[idx]['label']
        image_path = os.path.join(self.images_path, str(image_name + ".jpg"))
        img = Image.open(image_path).convert("RGB")

        # img = transforms.ToTensor()(img)
        img = transforms.Resize((32, 32))(img)

        if self.transform:
            img = self.transform(img)

        return [img, label]
            
