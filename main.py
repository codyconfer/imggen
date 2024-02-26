import io
import os
from typing import Optional
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response
from PIL import Image
from torch import autocast
from diffusers import StableDiffusionPipeline
from time import time
from slugify import slugify

# If I wasn't lazy, these could be env vars
model_org = "runwayml"
model_name = "stable-diffusion-v1-5"
reps = 1
width = 1024
height = 1024

# check for CUDA GPU
if not torch.cuda.is_available():
    raise Exception("Where CUDA bruh?")

torch.cuda.empty_cache()
print(torch.cuda.memory_summary(device=None, abbreviated=False))

#Calculate model storage path
model_path = f"models/{model_name}"


# download model
def download_model():
    model_id = f"{model_org}/{model_name}"
    model_download = StableDiffusionPipeline.from_pretrained(
        model_id,
        variant="fp16",
        torch_dtype=torch.float16,
    )
    model_download.save_pretrained(model_path)

if not os.path.exists(model_path):
    download_model()


# load model into VRAM
def load_model():
    try:
        m = StableDiffusionPipeline.from_pretrained(
            model_path, safety_checker=None, variant="fp16", torch_dtype=torch.float16
        )
    except:
        print("yeah yeah, dirty dirty...")
    m.to("cuda")
    m.enable_attention_slicing()
    print(m.device)
    print(torch.cuda.memory_summary(device=None, abbreviated=False))
    return m


# query model with prompt
async def query_model(prompt: str, m):
    files = []
    for i in range(reps):
        with autocast("cuda"):
            output = m(
                prompt,
                generator=torch.Generator("cuda").manual_seed(42),
                num_inference_steps=50,  # diffusion iterations
                guidance_scale=7.5,  # adherence to text, default 7.5
                width=width,
                height=height,
            )
        image = output.images[0]
        filename = f"images/{time()}_{slugify(prompt[:100])}.png"
        image.save(filename)
        files.append(filename)
    return files

# setup model and fastapi
model = load_model()
app = FastAPI()

# input model
class PromptInput:
    prompt: str


# endpoint
@app.post("/prompt")
async def post_prompt(input: PromptInput):
    if not input and not input.prompt:
        raise HTTPException(status_code=400, detail="No prompt received")
    try:
        files = await query_model(input.prompt, model)
        img = Image.open(files[0])
        byte_io = io.BytesIO()
        img.save(byte_io, "PNG")
        byte_io.seek(0)
        return Response(content=byte_io.read(), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
