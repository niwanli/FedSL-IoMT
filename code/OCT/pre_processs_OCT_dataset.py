import os
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image
import torch
import torchvision
import numpy as np
from PIL import Image
import pandas as pd

info_array = []  # A 3-dimensional array with three dimensions: file path, file name, and category label (numeric)
col = ['image_id', 'label']

dataset_dir = '/Users/86188/Desktop/data'
# Read multiple categories in the file path (one folder per category)
classes = os.listdir(dataset_dir)
print("image classes length:", len(classes))
for kindname in classes:
    # Gets the path to each category folder
    if (kindname.startswith('.')):
        print("pass .DStore file")
    else:
        classpath = dataset_dir + '/' + kindname
        for image_id in os.listdir(classpath):
            # Read the path information for each image file in the folder of each class
            label =kindname  # Converts the string label of label to a numeric label
            info_array.append([image_id, label])

info_array = np.array(info_array)
info_array.shape
# print(info_array)

df = pd.DataFrame(info_array, columns=col)
# df.to_csv("/Users/86188/Desktop/chest_xray_dataset.csv", encoding='UTF-8')
df.to_csv("/Users/86188/Desktop/OCT_metadata.csv", encoding='UTF-8')