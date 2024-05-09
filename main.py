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

from src.utils import visualize_map, SegmentationDataset, collate_fn, id2label
from src.dino import Dinov2ForSemanticSegmentation

# Load dataset from Hugging Face
ds = load_dataset("vinczematyas/stranger_sections_2")
# trim dataset to 10 samples for each split
# ds = {split: ds[split].select(range(100)) if split=="unlabeled" else ds[split] for split in ds.keys()}

# Visualize a random image
# idx = np.random.randint(0, len(ds["train"]))
# visualize_map(ds["train"][idx]["image"], np.array(ds["train"][idx]["segmentation"]))

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

model = Dinov2ForSemanticSegmentation.from_pretrained("facebook/dinov2-base", id2label=id2label, num_labels=len(id2label))

for name, param in model.named_parameters():
    if name.startswith("dinov2"):
        param.requires_grad = False

optimizer = AdamW(model.parameters(), lr=5e-5)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

metric = evaluate.load("mean_iou")
model.train()

for epoch in trange(25, desc="Initial training"):
    for idx, batch in enumerate(train_dl):
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

'''
for epoch in trange(5, desc="Pseudo-labeling"):
    pseudo_labels = []
    accepted = []

    for idx, batch in enumerate(unlabeled_dl):
        pixel_values = batch["pixel_values"].to(device)

        # forward pass
        outputs = model(pixel_values)

        upsampled_logits = torch.nn.functional.interpolate(
            outputs.logits,
            size=(1024, 1360), 
            mode="bilinear",
            align_corners=False
        )

        # thresholding
        predicted = upsampled_logits.argmax(dim=1)

        threshold = 0.8

        acceptance = upsampled_logits.max(dim=1).values > threshold
        predicted[~acceptance] = 0

        pseudo_labels.append(predicted.cpu().numpy())

    unlabeled_ds = {
        "image": unlabeled_ds.dataset["image"], 
        "segmentation": [Image.fromarray(np.uint8(i.squeeze())) for i in pseudo_labels],
        "file_name": unlabeled_ds.dataset["file_name"]
    }
    unlabeled_ds = SegmentationDataset(
        Dataset.from_dict(unlabeled_ds),
        train_transform
    )

    combined_ds = SegmentationDataset(
        concatenate_datasets([train_ds.dataset, unlabeled_ds.dataset]),
        train_transform
    )
    combined_dl = DataLoader(combined_ds, batch_size=1, shuffle=True, collate_fn=collate_fn)

    for epoch in trange(5, desc="Pseudo-labeling training"):
        print("Epoch:", epoch)
        for idx, batch in enumerate(combined_dl):
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

            if (idx+1) % 100 == 0:
                metrics = metric.compute(
                    num_labels=len(id2label),
                    ignore_index=0,
                    reduce_labels=False,
                )
                print("Loss:", loss.item())
                print("Mean_iou:", metrics["mean_iou"])
                print("Mean accuracy:", metrics["mean_accuracy"])
'''

model.eval()
with torch.no_grad():
    for idx, batch in enumerate(test_dl):
        pixel_values = batch["pixel_values"].to(device)

        # forward pass
        outputs = model(pixel_values)

        upsampled_logits = torch.nn.functional.interpolate(
            outputs.logits,
            size=(1024, 1360), 
            mode="bilinear",
            align_corners=False
        )

        predicted = upsampled_logits.argmax(dim=1)

        np.save(f"predictions/{batch['file_names'][0]}_pred.npy", predicted.cpu().numpy())

