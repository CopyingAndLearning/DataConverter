"""
name：code & test
time：2024/11/20 19:40
author：yxy
content：
"""

import os
img_list = []
path = "./image"
for i in os.listdir(path):
    img_list.append(path + "/" + i)
print(img_list)

from datasets import Dataset, Image
dataset = Dataset.from_dict({"image": img_list}).cast_column("image", Image())
print(dataset[0])
dataset2 = Dataset.from_dict({"image2": img_list}).cast_column("image", Image())

from datasets import concatenate_datasets
merged_dataset = concatenate_datasets([dataset, dataset2])
print(merged_dataset)

