import torch
from PIL import Image
import clip
import faiss
import numpy as np
import json

from config import CACHE_FOLDER, DATASET_FOLDER

cache_file = CACHE_FOLDER+"vector.index"

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

index = faiss.read_index(cache_file)

with open(CACHE_FOLDER+"files.json", "r") as file:
    textos = json.load(file)

def search_vector(vector, K):
    vector = vector.detach().cpu().numpy()
    vector = np.float32(vector)
    #faiss.normalize_L2(vector)
    distances, indices = index.search(vector.astype('float32'), K)
    results = [[textos[v], float(distances[0][k])] for k, v in enumerate(indices[0])]   # float32 not compatible with json
    return results

# Función para procesar el archivo con PyTorch
def process_image(file, K = 6):
    image_selected = preprocess(Image.fromarray(file)).unsqueeze(0).to(device)
    master_image = model.encode_image(image_selected)

    return search_vector(master_image, K)

# Función para procesar el texto con PyTorch
def process_text(texto, K = 6):
    text = clip.tokenize([texto]).to(device)
    text_features = model.encode_text(text)
    
    return search_vector(text_features, K)

# Función para procesar el texto vs imagen con PyTorch
def process_image_text(texto, file):
    sentences = [s.strip() for s in texto.split(",")]
    text_token = clip.tokenize(sentences).to(device)
    text_features = model.encode_text(text_token)
    image_selected = preprocess(Image.fromarray(file)).unsqueeze(0).to(device)
    master_image = model.encode_image(image_selected)

    text_features = torch.nn.functional.normalize(text_features, dim=1, p=2)
    master_image = torch.nn.functional.normalize(master_image, dim=1, p=2)

    results = []
    cos = torch.nn.CosineSimilarity(dim=0)
    #For each image, compute its cosine similarity with the prompt and store the result in a dict
    for index, text_f in enumerate(text_features):
        with torch.no_grad():
            sim = cos(text_f,master_image[0]).item()
            sim = (sim+1)/2
            results.append([sentences[index],sim])

    return results