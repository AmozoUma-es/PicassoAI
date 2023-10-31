# coding=utf-8

import gradio as gr
from process import process_image, process_text, process_image_text

def predict(texto, image, K):
    if isinstance(texto, str) and texto.strip() != "": # Si el texto no es vacio
        if image is not None:
            results = process_image_text(texto, image)
            results = [f'({x[0]}: {x[1]})' for x in results]
            return [
                (image, x) for x in results
            ]
        else:
            results = process_text(texto, K)
    elif image is not None:
        results = process_image(image, K)
    return [
        (x[0], f'{x[1]:.8f}') for x in results
    ]

salida = gr.Interface(
    title="PicassoAI",
    description = "Creado por Alejando Mozo usando CLIP de OpenAI",
    fn=predict,
    inputs=[
        gr.Textbox(label="Prompt"),
        gr.Image(label="Image"),
        gr.Slider(minimum=1, maximum=12, default=6, step=1, label="Number of results")
    ],
    outputs=[
        gr.Gallery( label="Images", show_label=False, elem_id="gallery", columns=[2], rows=[2], object_fit="contain", height="auto")  # Muestra las imagenes
    ],
    css="footer{display:none !important}"
)

salida.launch()
