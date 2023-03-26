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
    def __init__(self, annotations_file, img_dir):
        self.img_dir = img_dir

        with open(annotations, 'r') as f:
            annotations = json.load(f, annotations)
        
        annotations = pd.read_csv(annotations_file)
        def transform_center(x,y,w,h):
            """from x_min, y_min to center"""
            return [x + w // 2, y + h // 2, w,h]
        annotations.bbox = [transform_center(*eval(x)) for x in annotations.bbox]

        self.bboxes = annotations.groupby('image_id')['bbox'].apply(lambda x: torch.Tensor(list(x)))
        self.images = np.unique(annotations['image_id'])

        self.transform = T.Compose([
            T.ToTensor(),
            # T.Resize(800),
            # T.Normalize([0.3146, 0.3174, 0.2144],[0.4644, 0.4655, 0.4104])
        ])
    
    def __len__(self):
        return len(self.images)

    def get_img(self, idx):
        img_id = self.images[idx]
        img_path = os.path.join(self.img_dir, f"{img_id}.jpg")
        image = Image.open(img_path)
        
        return image

    def __getitem__(self, idx):
        img_id = self.images[idx]
        img_path = os.path.join(self.img_dir, f"{img_id}.jpg")
        image = Image.open(img_path)
        image = self.transform(image)
        
        bboxes = self.bboxes[img_id]
        # must scale bboxes to be within the image
        # print(image.shape)
        _,X,Y = image.shape
        scale = torch.Tensor([1/X,1/Y,1/X,1/Y])
        bboxes = torch.einsum('ij,j->ij',bboxes, scale)

        # for now, just put the label as 1
        labels = torch.ones(len(bboxes), dtype=torch.long)
        return image, {'labels': labels, 'boxes': bboxes}

