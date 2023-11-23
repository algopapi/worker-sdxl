# builder/model_fetcher.py

import torch
from diffusers import (ControlNetModel,
                       StableDiffusionXLControlNetInpaintPipeline,
                       StableDiffusionXLImg2ImgPipeline,
                       StableDiffusionXLPipeline)


def fetch_pretrained_model(model_class, model_name, **kwargs):
    '''
    Fetches a pretrained model from the HuggingFace model hub.
    '''
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

def fetch_pretrained_model_with_controlnet(model_class, model_name, controlnet, **kwargs):
    '''
    Fetches a pretrained model from the HuggingFace model hub.
    '''
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

def get_diffusion_pipelines():
    '''
    Fetches the Stable Diffusion XL pipelines from the HuggingFace model hub.
    '''
    common_args = {
        "torch_dtype": torch.float16,
        "variant": "fp16",
        "use_safetensors": True
    }
    controlnet = fetch_pretrained_model(ControlNetModel, "diffusers/controlnet-canny-sdxl-1.0",
                                        **common_args)
    sdxl_contorlnet_inpaint_pipe = fetch_pretrained_model_with_controlnet(StableDiffusionXLControlNetInpaintPipeline,
                                                                            "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
                                                                            controlnet,
                                                                            **common_args)
                                                          
    sdxl_inpaint_pipe = fetch_pretrained_model(StableDiffusionXLPipeline,
                                  "stabilityai/stable-diffusion-xl-base-1.0", **common_args)
    # refiner = fetch_pretrained_model(StableDiffusionXLImg2ImgPipeline,
    #                                  "stabilityai/stable-diffusion-xl-refiner-1.0", **common_args)

    return sdxl_inpaint_pipe, sdxl_contorlnet_inpaint_pipe


if __name__ == "__main__":
    get_diffusion_pipelines()
