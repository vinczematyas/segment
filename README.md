# TODO

- [ ] Ash try to push/pull
- [ ] data augmentation
- [ ] DINO segmentation backbone + simple classification head
- [ ] FeatUp to scale the segmentation

## folder structure

    .
    ├── data
    │   ├── train
    │   │   ├── image
    │   │   ├── label
    │   ├── test
    │   │   ├── image
    │   ├── unlabeled
    │   │   ├── image
    ├── dev.py
    ├── README.md

## current pipeline

1. data augmentation
2. ¿DINO? segmentation backbone (frozen or fine-tuned using an adapter)
3. FeatUp to scale the segmentation
4. Simple classification head using the segmentation features
