import numpy as np
from typing import Literal
import matplotlib.pyplot as plt
import PIL.Image as Image
from glob import glob

# from notebook
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    batch_size: int = 8
    epochs: int = 1
    learning_rate: float = 1e-4
    lr_decay_rate: float = 0.9998
    seed: int = 420
    model_name: str = 'nvidia/mit-b0'
    project_name: str = 'segment'  # for wandb
    device: Optional[str] = field(default_factory=lambda: 'cuda' if torch.cuda.is_available() else 'cpu')

    def asdict(self):
        return vars(self)

cfg = TrainingConfig()

def load_train_data(image_folder, segmentation_folder):
    image_segmentation_pairs = {"image": [], "segmentation": []}
    for filename in os.listdir(image_folder):
        if filename.endswith(".JPG"):
            image_path = os.path.join(image_folder, filename)
            segmentation_path = os.path.join(segmentation_folder, filename.replace(".JPG", "_gt.npy"))

            image = Image.open(image_path)
            segmentation = np.load(segmentation_path)

            image_segmentation_pairs["image"].append(image)
            image_segmentation_pairs["segmentation"].append(segmentation)

    dataset = Dataset.from_dict(image_segmentation_pairs)
    return dataset

image_folder = "data/train/image"
segmentation_folder = "data/train/label"

ds = load_train_data(image_folder, segmentation_folder)
ds.shuffle(seed=cfg.seed)

id2label = {
    "0": "lipnite",
    "1": "vitrinite",
    "2": "inertinite"
}
label2id = {v: k for k, v in id2label.items()}

tokenizer = SegformerImageProcessor.from_pretrained(cfg.model_name)
model = AutoModelForImageSegmentation.from_pretrained(
    cfg.model_name, num_labeles = 3, id2label=id2label, label2id=label2id
).to(cfg.device)

# Image Transformations
img_transforms = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.OneOf(
            [
                A.Downscale(p=0.1, scale_min=0.4, scale_max=0.6),
                A.GaussNoise(p=0.2),
            ],
            p=0.1,
        ),
        A.OneOf(
            [
                A.RandomBrightnessContrast(p=0.2),
                A.ColorJitter(p=0.2),
                A.HueSaturationValue(p=0.2),
            ],
            p=0.1,
        ),
        A.OneOf([A.PixelDropout(p=0.2), A.RandomGravel(p=0.2)], p=0.15),
    ]
)

train_ds = None  # TODO: transform dataset using set_trainsform
train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)

# TODO: continue w. training loop from https://github.com/mattmdjaga/segformer_b2_clothes/blob/main/train_book.ipynb
