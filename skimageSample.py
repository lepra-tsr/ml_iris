import skimage
import os
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, filters, io

camera = data.camera()

print(type(camera), camera.shape)

coins = data.coins()
threshold_value = filters.threshold_otsu(coins)
print(threshold_value)

project_root = os.path.dirname(os.path.abspath(__file__))
filename = os.path.join(project_root, 'sampleImage/olympus_cat.jpg')
cat = io.imread(filename)
print(type(cat), cat.shape, len(cat))  # 225, 300, 3

# xy(2次元) + rgb(+1次元) の3次元データを、xyへマッピング
red_2d = np.apply_along_axis(lambda x: x[0], 2, cat)

# @SEE https://note.nkmk.me/python-seaborn-heatmap/
plt.figure(figsize=(7, 4))
sns.heatmap(red_2d)
plt.imshow(cat)
plt.show()
