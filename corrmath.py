import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset, Dataset, concatenate_datasets
import torch
from torch.utils.data import DataLoader
import albumentations as A
from typing import Optional
from torch.optim import AdamW
from tqdm import trange
import evaluate
from PIL import Image
from peft import LoraConfig

from src.utils import visualize_map, SegmentationDataset, collate_fn, id2label, print_trainable_parameters
from src.dino import Dinov2ForSemanticSegmentation

# Load dataset from Hugging Face
ds = load_dataset("vinczematyas/stranger_sections_2")

# Define the transforms
train_transform = A.Compose([
    A.Resize(448, 448),  # TODO: what shoudl be the size?
    A.Normalize(mean=(0.443, 0.366, 0.230), std=(0.213, 0.211, 0.219)),  # TODO: change from ImageNet mean and std
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    # TODO: add more augmentations
])
test_transform = A.Compose([
    A.Resize(448, 448),
    A.Normalize(mean=(0.466, 0.385, 0.194), std=(0.216, 0.220, 0.181)),
])

# Create the datasets
train_ds = SegmentationDataset(ds["train"], train_transform)
unlabeled_ds = SegmentationDataset(ds["unlabeled"], train_transform)
test_ds = SegmentationDataset(ds["test"], test_transform)

# Create the dataloaders
train_dl = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=collate_fn)
unlabeled_dl = DataLoader(unlabeled_ds, batch_size=1, shuffle=True, collate_fn=collate_fn)
test_dl = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["query", "value"],
    lora_dropout=0.1,
    bias="none",
)

model = Dinov2ForSemanticSegmentation.from_pretrained(
    "facebook/dinov2-base", 
    id2label=id2label, 
    num_labels=len(id2label),
    lora_config=lora_config)

print_trainable_parameters(model.dinov2)
print_trainable_parameters(model.classifier)
print_trainable_parameters(model)

