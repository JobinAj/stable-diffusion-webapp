from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from diffusers import DiffusionPipeline
import torch
import uuid
import os

# Initialize FastAPI app
app = FastAPI()

# Load Stable Diffusion model
pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to("cuda" if torch.cuda.is_available() else "cpu")

# Ensure static directory exists
os.makedirs("static", exist_ok=True)

class TextPrompt(BaseModel):
    prompt: str

@app.post("/generate/")
def generate_image(prompt: TextPrompt):
    try:
        image = pipe(prompt.prompt).images[0]
        filename = f"static/{uuid.uuid4()}.png"
        image.save(filename)
        return {"image_url": f"/{filename}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def home():
    return {"message": "Stable Diffusion API is running!"}
