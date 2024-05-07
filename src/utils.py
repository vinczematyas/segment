import numpy as np
import matplotlib.pyplot as plt
import torch


# Define the collate function
def collate_fn(inputs):
    batch = dict()
    batch["pixel_values"] = torch.stack([i[0] for i in inputs], dim=0)
    batch["labels"] = torch.stack([i[1] for i in inputs], dim=0)
    batch["original_images"] = [i[2] for i in inputs]
    batch["original_segmentation_maps"] = [i[3] for i in inputs]
    return batch


# Define the dataset class
class SegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        original_image = np.array(item["image"])
        original_segmentation_map = np.array(item["segmentation"])

        transformed = self.transform(image=original_image, mask=original_segmentation_map)
        image, target = torch.tensor(transformed["image"]), torch.tensor(transformed["mask"])

        image = image.permute(2, 0, 1)

        return image, target, original_image, original_segmentation_map


id2color = {k: list(np.random.choice(range(256), size=3)) for k,v in id2label.items()}
id2color = {
    1: [74, 167, 79],
    2: [253, 151, 15],
    3: [234, 73, 71]
}

# Visualize the dataset
def visualize_map(image, segmentation_map):
    color_seg = np.zeros((segmentation_map.shape[0], segmentation_map.shape[1], 3), dtype=np.uint8) # height, width, 3
    for label, color in id2color.items():
        color_seg[segmentation_map == label, :] = color

    # Show image + mask
    img = np.array(image) * 0.5 + color_seg * 0.5
    img = img.astype(np.uint8)

    plt.figure(figsize=(15, 10))
    plt.imshow(img)
    plt.show()
