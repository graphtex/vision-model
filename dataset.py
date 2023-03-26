import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.ops import box_convert
from torchvision.utils import draw_bounding_boxes
import torchvision.transforms as T
import os
import pandas as pd
import numpy as np
from PIL import Image
import json

class GraphImageDataSet(Dataset):
    def __init__(self, annotations_file, img_dir, filter_nodes = False):
        self.img_dir = img_dir
        self.filter_nodes = filter_nodes

        with open(annotations_file, 'r') as f:
            annotations = json.load(f)
        
        self.annotations = annotations
        # self.bboxes = annotations.groupby('image_id')['bbox'].apply(lambda x: torch.Tensor(list(x)))
        self.images = np.array(list(annotations.keys()))

        self.transform = T.Compose([
            T.ToTensor(),
            T.Resize(800),
            # T.Normalize([0.3146, 0.3174, 0.2144],[0.4644, 0.4655, 0.4104])
        ])
    
    def __len__(self):
        return len(self.images)

    def get_img(self, idx):
        img_id = self.images[idx]
        img_path = os.path.join(self.img_dir, f"{img_id}.png")
        image = Image.open(img_path).convert('RGB')
        
        return image

    def __getitem__(self, idx):
        img_id = self.images[idx]
        img_path = os.path.join(self.img_dir, f"{img_id}.png")
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        
        # bboxes = self.bboxes[img_id]
        # must scale bboxes to be within the image
        # print(image.shape)
        # _,X,Y = image.shape
        # scale = torch.Tensor([1/X,1/Y,1/X,1/Y])
        # bboxes = torch.einsum('ij,j->ij',bboxes, scale)

        # for now, just put the label as 1
        # labels = torch.ones(len(bboxes), dtype=torch.long)
        labels = torch.Tensor(self.annotations[img_id]['labels']).to(torch.int64)
        boxes = torch.Tensor(self.annotations[img_id]['boxes'])

        valid_labels = labels > 35
        labels = labels[valid_labels]
        boxes = boxes[valid_labels]

        if self.filter_nodes:
            valid_labels = labels != 36
            labels = labels[valid_labels]
            boxes = boxes[valid_labels]

        return image, {'labels': labels, 'boxes': boxes}

class GraphTestImages(Dataset):
    def __init__(self, img_dir) -> None:
        self.img_dir = img_dir
        self.transform = T.Compose([T.ToTensor(),T.Resize(800)])
        self.images = os.listdir(self.img_dir)

    def __len__(self):
      return len(self.images)
    
    def __getitem__(self, idx):
        img_id = self.images[idx]
        img_path = os.path.join(self.img_dir, img_id)
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        
        return image
