# coding=utf-8

import gradio as gr
import torch
from PIL import Image
import clip
import faiss
import numpy as np
import json

cache_file = 'cache/vector.index'
K = 6

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

index = faiss.read_index(cache_file)

with open("cache/files.json", "r") as file:
    textos = json.load(file)

def search_vector(vector):
    vector = vector.detach().cpu().numpy()
    vector = np.float32(vector)
    #faiss.normalize_L2(vector)
    distances, indices = index.search(vector.astype('float32'), K)
    results = [[textos[v], float(distances[0][k])] for k, v in enumerate(indices[0])]   # float32 not compatible with json
    return results

# Función para procesar el archivo con PyTorch
def procesar_archivo(file):
    image_selected = preprocess(Image.fromarray(file)).unsqueeze(0).to(device)
    master_image = model.encode_image(image_selected)

    return search_vector(master_image)

# Función para procesar el texto con PyTorch
def procesar_texto(texto):
    text = clip.tokenize([texto]).to(device)
    text_features = model.encode_text(text)
    
    return search_vector(text_features)

# Función para procesar el texto vs imagen con PyTorch
def procesar_imagen_texto(texto, file):
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

    return '\n<br>'.join([f'({x[0]}: {x[1]})' for x in results])

def predict(texto, image):
    if isinstance(texto, str) and texto.strip() != "": # Si el texto no es vacio
        if image is not None:
            print('imagen y texto')
            results = procesar_imagen_texto(texto, image)
            return [
                gr.Image(image),  # Muestra la imagen
                gr.Label(f'{results}')  # Muestra el valor
            ]
        else:
            print("texto")
            results = procesar_texto(texto)
    elif image is not None:
        print("imagen")
        results = procesar_archivo(image)
    return [
        gr.Image(results[0][0]),  # Muestra la imagen
        gr.Label(f'{results[0][1]:.8f}')  # Muestra el valor
    ]

salida = gr.Interface(
    title="PicassoAI",
    description = "Creado por Alejando Mozo usando CLIP de OpenAI",
    fn=predict,
    inputs=[
        gr.Textbox(label="Prompt"),
        gr.Image(label="Image")
    ],
    outputs=[
        gr.Image(label="Image"),  # Muestra la imagen
        gr.Label(label="Cosine similarity")  # Muestra el valor
    ],
    css="footer{display:none !important}"
)

salida.launch()
