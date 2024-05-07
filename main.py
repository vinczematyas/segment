import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
import albumentations as A
from typing import Optional
from torch.optim import AdamW
from tqdm.auto import tqdm
import evaluate

from src.utils import visualize_map, SegmentationDataset, collate_fn, id2label
from src.dino import Dinov2ForSemanticSegmentation

# Load dataset from Hugging Face
ds = load_dataset("vinczematyas/stranger_sections_2")

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

epochs = 1
optimizer = AdamW(model.parameters(), lr=5e-5)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

evaluate.list_evaluation_modules
assert False

metric = evaluate.load("mean_iou")
model.train()

for epoch in range(epochs):
    print("Epoch:", epoch)
    for idx, batch in enumerate(tqdm(train_dl)):
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

    # forward pass
    outputs = model(pixel_values, labels=labels)
    loss = outputs.loss

    loss.backward()
    optimizer.step()

    # zero the parameter gradients
    optimizer.zero_grad()

    # evaluate
    with torch.no_grad():
        predicted = outputs.logits.argmax(dim=1)

        # note that the metric expects predictions + labels as numpy arrays
        metric.add_batch(predictions=predicted.detach().cpu().numpy(), references=labels.detach().cpu().numpy())

    if idx % 100 == 0:
        metrics = metric.compute(
            num_labels=len(id2label),
            ignore_index=0,
            reduce_labels=False,
        )

        print("Loss:", loss.item())
        print("Mean_iou:", metrics["mean_iou"])
        print("Mean accuracy:", metrics["mean_accuracy"])



