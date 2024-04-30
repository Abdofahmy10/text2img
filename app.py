import streamlit as st
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image, ImageDraw

cartoon_checkpoint = 'redstonehero/cetusmix_v4'
RealVisXL_V4_checkpoint = 'SG161222/RealVisXL_V4.0'
clip_checkpoint = "runwayml/stable-diffusion-v1-5"

real = StableDiffusionPipeline.from_pretrained(RealVisXL_V4_checkpoint, torch_dtype=torch.float16)
real = real.to("cuda")
real.safety_checker = None

cartoon = StableDiffusionPipeline.from_pretrained(cartoon_checkpoint, torch_dtype=torch.float16)
cartoon = cartoon.to("cuda")
cartoon.safety_checker = None

clip = StableDiffusionPipeline.from_pretrained(clip_checkpoint, torch_dtype=torch.float16)
clip = clip.to("cuda")
clip.safety_checker = None

def generate_image(prompt:str, height=800, width=800, model="real"):
    steps = 50
    guidance = 7
    neg = "easynegative, human, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worstquality, low quality, normal quality, jpegartifacts, signature, watermark, username, blurry, bad feet, cropped, poorly drawn hands, poorly drawn face, mutation, deformed, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, extra fingers, fewer digits, extra limbs, extra arms, extra legs, malformed limbs, fused fingers, too many fingers, long neck, cross-eyed, mutated hands, polar lowres, bad body, bad proportions, gross proportions, text, error, missing fingers, missing arms, missing legs, extra digit, extra arms, extra leg, extra foot,"
    
    if model == "real":
        pipeline = real
    elif model == "cartoon":
        pipeline = cartoon
    elif model == "clip":
        pipeline = clip
    else:
        st.error("Invalid model selected.")
        return
    
    image = pipeline(prompt, height=height, width=width, num_inference_steps=steps, guidance_scale=guidance, negative_prompt=neg).images[0]
    return image

def main():
    # Set up sidebar
    st.sidebar.title("Diffusers ♥")
    model = st.sidebar.selectbox("Select your model", ("real", "cartoon", "clip"))

    if model == "real":
        render_text2img(model)
    elif model == "cartoon":
        render_text2img(model)
    elif model == "clip":
        render_text2img(model)

def render_text2img(model):
    # Display the input controls for Text2Img task
    st.title("Text2Img ♥ 🚀")
    text = st.text_area("Enter the text")
    height = st.number_input("Enter the height", value=800)
    width = st.number_input("Enter the width", value=800)
    generate_button = st.button("Generate")

    if generate_button:
        x = generate_image(text, height, width, model)
        st.image(caption='Generated Image', image=x)

if __name__ == "__main__":
    main()