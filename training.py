import json
import torch
import clip
import os
from PIL import Image
import time
import numpy as np
import faiss

from config import CACHE_FOLDER, DATASET_FOLDER

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
#my batch of images

time_start = time.perf_counter()

images = []
images_names = []
for root, dirs, files in os.walk(DATASET_FOLDER):
    for file in files:
        if file.endswith(("jpg","jpeg")):
            images_names.append(file)
            images.append(  root  + file)

time_reading = time.perf_counter()

cache_file = CACHE_FOLDER+"vector.index"

images_processed = 0

processed_images = {}
for img in images:
    with torch.no_grad():
        image_preprocess = preprocess(Image.open(img)).unsqueeze(0).to(device)
        image_features = model.encode_image( image_preprocess)
        processed_images[img] = image_features
    images_processed += 1

vectors = [item.detach().cpu().numpy() for item in processed_images.values()]
files_names = list(processed_images.keys())
vectors_np = np.array(vectors)
vectors_np = vectors_np.reshape(vectors_np.shape[0], -1) ## Flatten

print(vectors_np.shape)

time_processing = time.perf_counter()

# FlatIP for cosine similarity
index = faiss.IndexFlatIP(vectors_np.shape[1])

faiss.normalize_L2(vectors_np)
index.add(vectors_np.astype('float32'))

#Store the index locally
faiss.write_index(index, cache_file)
#Store the index of files
with open(CACHE_FOLDER+"files.json", "w") as file:
    json.dump(files_names, file)

print(f'Images processed: {images_processed} reading: {time_reading - time_start} processing: {time_processing - time_reading} processing: {time_processing - time_reading}')
