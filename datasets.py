import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import json
import torchvision.transforms as transforms

class CocoDetectionDataset(Dataset):
    def __init__(self, root_dir, annotation_file, transform=None):
        """
        Args:
            root_dir (string): Root directory of the COCO dataset.
            annotation_file (string): Path to the COCO annotation file (e.g., "instances_train2017.json").
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.annotations = []
        
        # Load the COCO annotation file
        with open(annotation_file, 'r') as f:
            coco_data = json.load(f)
            self.categories = coco_data['categories']
            self.annotations = coco_data['annotations']
            
        # Create a mapping from category_id to index
        self.category_id_to_index = {cat['id']: idx for idx, cat in enumerate(self.categories)}

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, 'train2017', self.annotations[idx]['file_name'])
        image = Image.open(img_name).convert("RGB")
        
        boxes = self.annotations[idx]['bbox']  # Format: [x, y, width, height]
        category_id = self.annotations[idx]['category_id']
        label = self.category_id_to_index[category_id]

        # Apply transforms if available
        if self.transform:
            image = self.transform(image)
        
        # Convert box coordinates to (x_min, y_min, x_max, y_max) format
        box = torch.tensor([boxes[0], boxes[1], boxes[0] + boxes[2], boxes[1] + boxes[3]], dtype=torch.float32)
        
        return image, {'boxes': box, 'labels': label}
