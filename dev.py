import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
import albumentations as A
from typing import Optional

from src.utils import visualize_map, SegmentationDataset, collate_fn
from src.dino import Dinov2ForSemanticSegmentation

# Load dataset from Hugging Face
ds = load_dataset("vinczematyas/stranger_sections_2")

# Define the labels
id2label = {
    0: "background",
    1: "lipnite",
    2: "vitrinite",
    3: "inertinite"
}

# Visualize a random image
# idx = np.random.randint(0, len(ds["train"]))
# visualize_map(ds["train"][idx]["image"], np.array(ds["train"][idx]["segmentation"]))

# Define the transforms
train_transform = A.Compose([
    A.Resize(256, 256),  # TODO: what shoudl be the size?
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))  # TODO: change from ImageNet mean and std
    # TODO: add more augmentations
])
test_transform = A.Compose([])

# Create the datasets
train_ds = SegmentationDataset(ds["train"], train_transform)
test_ds = SegmentationDataset(ds["test"], test_transform)

# Create the dataloaders
train_dl = DataLoader(train_ds, batch_size=2, shuffle=True, collate_fn=collate_fn)
test_dl = DataLoader(test_ds, batch_size=2, shuffle=False, collate_fn=collate_fn)

model = Dinov2ForSemanticSegmentation.from_pretrained("facebook/dinov2-base", id2label=id2label, num_labels=len(id2label))

for name, param in model.named_parameters():
  if name.startswith("dinov2"):
    param.requires_grad = False
print(model)
