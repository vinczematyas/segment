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
from peft import LoraConfig, get_peft_model

from src.utils import visualize_map, SegmentationDataset, collate_fn, id2label
from src.dino import Dinov2ForSemanticSegmentation

# Load dataset from Hugging Face
ds = load_dataset("vinczematyas/stranger_sections_2")

# Create the datasets
train_ds = SegmentationDataset(ds["train"], train_transform)
unlabeled_ds = SegmentationDataset(ds["unlabeled"], train_transform)
test_ds = SegmentationDataset(ds["test"], test_transform)

# Create the dataloaders
train_dl = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=collate_fn)
unlabeled_dl = DataLoader(unlabeled_ds, batch_size=1, shuffle=True, collate_fn=collate_fn)
test_dl = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

model = Dinov2ForSemanticSegmentation.from_pretrained("facebook/dinov2-base", id2label=id2label, num_labels=len(id2label))

config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["query", "value"],
    lora_dropout=0.1,
    bias="none",
    modules_to_save=["classifier"],
)
lora_model = get_peft_model(model, config)
print_trainable_parameters(lora_model)
