import numpy as np
from typing import Literal
import matplotlib.pyplot as plt
import PIL.Image as Image
from glob import glob

img = np.asarray(Image.open('./data/train/image/17gw5j.JPG'))
label = np.load('./data/train/label/17gw5j_gt.npy')

print(f'Shape of image: {img.shape}')
print(f'Shape of label: {label.shape}')

fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].imshow(img)
ax[0].set_title('Liptinite Image')
ax[0].axis('off')
ax[1].imshow(label)
ax[1].set_title('Liptinite Label')
ax[1].axis('off')
ax[2].imshow(img)
ax[2].set_title('Liptinite Image with Label')
ax[2].axis('off')
ax[2].imshow(label, alpha=0.5)
plt.show()

