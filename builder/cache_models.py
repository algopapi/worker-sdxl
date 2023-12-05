# builder/model_fetcher.py

import torch
from diffusers import (AutoencoderKL, ControlNetModel, DiffusionPipeline,
                       StableDiffusionXLControlNetInpaintPipeline,
                       StableDiffusionXLInpaintPipeline)


# FIX THIS
def fetch_vae(model_class, model_name, **kwargs):
    '''
    Fetches a pretrained model from the HuggingFace model hub.
    '''
    print("FETCHING PRETRAINED MODEL")
    max_retries = 3
    for attempt in range(max_retries):
        try:
            return model_class.from_pretrained(model_name, torch_dtype=torch.float16)
        except OSError as err:
            if attempt < max_retries - 1:
                print(
                    f"Error encountered: {err}. Retrying attempt {attempt + 1} of {max_retries}...")
            else:
                raise


def fetch_pretrained_model(model_class, 
                               model_name,
                               **kwargs):
    '''
    Fetches a pretrained model from the HuggingFace model hub.
    '''
    print("FETCHING PRETRAINED MODEL")
    max_retries = 3
    for attempt in range(max_retries):
        try:
            return model_class.from_pretrained(model_name, **kwargs)
        except OSError as err:
            if attempt < max_retries - 1:
                print(
                    f"Error encountered: {err}. Retrying attempt {attempt + 1} of {max_retries}...")
            else:
                raise

def fetch_pretrained_model_vae(model_class, 
                               model_name, 
                               vae, 
                               **kwargs):
    '''
    Fetches a pretrained model from the HuggingFace model hub.
    '''
    print("FETCHING PRETRAINED MODEL")
    max_retries = 3
    for attempt in range(max_retries):
        try:
            return model_class.from_pretrained(model_name, vae=vae, **kwargs)
        except OSError as err:
            if attempt < max_retries - 1:
                print(
                    f"Error encountered: {err}. Retrying attempt {attempt + 1} of {max_retries}...")
            else:
                raise

def fetch_pretrained_model_with_controlnet_vae(model_class, model_name, controlnet, vae, **kwargs):
    '''
    Fetches a pretrained model from the HuggingFace model hub.
    '''
    print("FETCHING PRETRAINED MODEL")
    max_retries = 3
    for attempt in range(max_retries):
        try:
            return model_class.from_pretrained(model_name, controlnet=controlnet, vae=vae, **kwargs)
        except OSError as err:
            if attempt < max_retries - 1:
                print(
                    f"Error encountered: {err}. Retrying attempt {attempt + 1} of {max_retries}...")
            else:
                raise


def get_diffusion_pipelines():
    '''
    Fetches the Stable Diffusion XL pipelines from the HuggingFace model hub.
    '''
   
    vae = fetch_vae(AutoencoderKL, "madebyollin/sdxl-vae-fp16-fix")
    
    common_args = {
        "torch_dtype": torch.float16,
        "variant": "fp16",
        "use_safetensors": True
    }

    sdxl_refiner_pipe = fetch_pretrained_model_vae(
        model_class=DiffusionPipeline,
        model_name="stabilityai/stable-diffusion-xl-refiner-1.0", 
        vae=vae,
        **common_args
    )
    
    controlnet = fetch_pretrained_model(
        model_class=ControlNetModel, 
        model_name="diffusers/controlnet-canny-sdxl-1.0",
        **common_args
    )

    sdxl_contorlnet_inpaint_pipe = fetch_pretrained_model_with_controlnet_vae(
        model_class=StableDiffusionXLControlNetInpaintPipeline,
        vae=vae,
        model_name="diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        controlnet=controlnet,
        **common_args
    )
                                                          
    return vae, sdxl_refiner_pipe, sdxl_contorlnet_inpaint_pipe, controlnet


if __name__ == "__main__":
    get_diffusion_pipelines()
