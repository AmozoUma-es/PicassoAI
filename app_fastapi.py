from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from process import process_image, process_text, process_image_text
from pydantic import BaseModel
import numpy as np
from PIL import Image
from io import BytesIO

from config import ALLOWED_ORIGINS

class TextInput(BaseModel):
    q: str

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post('/process-image')
async def request_image(image: UploadFile = File()):
    if not image:
        return JSONResponse(content={"message": "Imagen no enviada", "resultado": ""})

    request_object_content = await image.read()
    img = Image.open(BytesIO(request_object_content))

    result = process_image(np.array(img))
    return {"message": "Imagen procesada", "items": result}

@app.post('/process-text')
async def request_text(data: TextInput):
    result = process_text(data.q)
    return {"message": "Texto procesado", "items": result}

@app.post('/process-text-image')
async def request_text_image(q: str = Form(...), image: UploadFile = File()):
    if not image:
        return JSONResponse(content={"message": "Imagen no enviada", "resultado": ""})

    request_object_content = await image.read()
    img = Image.open(BytesIO(request_object_content))

    result = process_image_text(q, np.array(img))
    sorted_value = sorted(result, key=lambda x: x[1], reverse=True)
    sorted_res = dict(sorted_value)
    tops = list(sorted_res.items())
    return {"message": "Texto vs Imagen procesado", "items": tops}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=5000)
