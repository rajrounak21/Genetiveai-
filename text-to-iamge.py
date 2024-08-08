import torch
from diffusers import StableDiffusionPipeline
from key import HUGGINGFACE_API_KEY
import os
os.environ["HUGGINGFACEHUB_API_TOKEN"]=HUGGINGFACE_API_KEY
# Replace 'your_api_key_here' with your actual Hugging Face API key
model_id = "stabilityai/stable-diffusion-2"
pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=os.getenv('HUGGINGFACEHUB_API_TOKEN'))
pipe = pipe.to("cuda")  # Use GPU if available

def generate_image(prompt):
    with torch.autocast("cuda"):
        image = pipe(prompt)["sample"][0]
    return image

# Example usage
prompt = "A beautiful sunset over a mountain range"
image = generate_image(prompt)
image.show()