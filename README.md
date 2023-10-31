# PicassoAI
Picasso AI using OpenAI's CLIP

3 Versions:
Flask:      app_flask.py
FastAPI:    app_fast_api.py
Gradio:     app_gradio.py and app_gradio_gallery.py

All main processing functions are located in process.py
training.py created the FAISS index and image json from the images located in 'dataset-images' (DATASET_FOLDER in config.py)