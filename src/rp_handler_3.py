'''
Contains the handler function that will be called by the serverless.
'''

import base64
import concurrent.futures
import os

import cv2
import numpy as np
import runpod
import torch
from diffusers import (ControlNetModel, DDIMScheduler,
                       DPMSolverMultistepScheduler, EulerDiscreteScheduler,
                       LMSDiscreteScheduler, PNDMScheduler,
                       StableDiffusionXLControlNetInpaintPipeline,
                       StableDiffusionXLInpaintPipeline)
from diffusers.utils import load_image
from PIL import Image
from runpod.serverless.utils import rp_cleanup, rp_upload
from runpod.serverless.utils.rp_validator import validate

from rp_schemas import INPUT_SCHEMA

torch.cuda.empty_cache()

# ------------------------------- Model Handler ------------------------------ #
class ModelHandler:
    def __init__(self):
        self.sdxl_inpaint = None
        self.sdxl_canny_controlnet_inpaint = None
        self.load_models()
    
    def load_sdxl_inpaint(self):
        sdxl_outpaint_pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
            "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        ).to("cuda", silence_dtype_warnings=True)
        sdxl_outpaint_pipe.enable_xformers_memory_efficient_attention()
        return sdxl_outpaint_pipe
    
    def load_sdxl_canny_controlnet_inpaint(self):
        canny_controlnet = ControlNetModel.from_pretrained(
            "diffusers/controlnet-canny-sdxl-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        )

        sdxl_controlnet_outpaint_pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
            "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
            controlnet=canny_controlnet,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        ).to("cuda", silence_dtype_warnings=True)

        sdxl_controlnet_outpaint_pipe.enable_xformers_memory_efficient_attention()
        return sdxl_controlnet_outpaint_pipe
    
    def load_models(self):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_sdxl_inpaint = executor.submit(self.load_sdxl_inpaint)
            future_sdxl_inpaint_canny_controlnet = executor.submit(self.load_sdxl_canny_controlnet_inpaint) # << fix it hre
            
            self.sdxl_inpaint = future_sdxl_inpaint.result()
            self.sdxl_canny_controlnet_inpaint = future_sdxl_inpaint_canny_controlnet.result() # < and here

MODELS = ModelHandler()

# ---------------------------------- Helper ---------------------------------- #
def _save_and_upload_images(images, job_id):
    os.makedirs(f"/{job_id}", exist_ok=True)
    image_urls = []
    for index, image in enumerate(images):
        image_path = os.path.join(f"/{job_id}", f"{index}.png")
        image.save(image_path)

        if os.environ.get('BUCKET_ENDPOINT_URL', False):
            image_url = rp_upload.upload_image(job_id, image_path)
            image_urls.append(image_url)
        else:
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode("utf-8")
                image_urls.append(f"data:image/png;base64,{image_data}")

    rp_cleanup.clean([f"/{job_id}"])
    return image_urls


def make_scheduler(name, config):
    return {
        "PNDM": PNDMScheduler.from_config(config),
        "KLMS": LMSDiscreteScheduler.from_config(config),
        "DDIM": DDIMScheduler.from_config(config),
        "K_EULER": EulerDiscreteScheduler.from_config(config),
        "DPMSolverMultistep": DPMSolverMultistepScheduler.from_config(config),
    }[name]


def create_canny_edge_image(image_input: Image.Image):
    """
    Takes an image, applies Canny edge detection, and returns the resulting image.

    Args:
    image_input (bytes): The input image in bytes format.

    Returns:
    bytes: The Canny edge-detected image in bytes format.
    """
     # Convert image bytes to numpy array
    image_array = np.array(image_input)
    grayscale_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    grayscale_image = cv2.convertScaleAbs(grayscale_image)
    image = cv2.Canny(grayscale_image, 100, 200)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    return Image.fromarray(image)

def sdxl_multi_step_inpaint(job_input: INPUT_SCHEMA):
    step_1_prompt = job_input['step_1_prompt']
    step_2_prompt = job_input['step_2_prompt']

    if step_2_prompt is None:
        step_2_prompt = step_1_prompt

    image = job_input['image_url']
    mask = job_input['mask_url']

    if job_input['seed'] is None:
        job_input['seed'] = int.from_bytes(os.urandom(2), "big")

    generator = torch.Generator("cuda").manual_seed(job_input['seed'])

    image = load_image(image).convert("RGB")
    mask = load_image(mask).convert("RGB")

    image.save("original_image.png")
    mask.save("original_mask.png")

    first_step_image = MODELS.sdxl_inpaint(
        strength=1.0,
        prompt=step_1_prompt,
        image=image,
        mask_image=mask,
        width=job_input['width'],
        height=job_input['height'],
        negative_prompt=job_input['step_1_negative_prompt'],
        prompt_2=job_input['step_1_prompt_2'],
        negative_prompt_2=job_input['step_1_negative_prompt_2'],
        num_inference_steps=job_input['step_1_num_inference_steps'],
        guidance_scale=job_input['step_1_guidance_scale'],
        num_images_per_prompt=job_input['step_1_num_images'],
        generator=generator,
    ).images[0]
    
    first_layer_canny_edge = create_canny_edge_image(first_step_image)

    second_step_outpaint = MODELS.sdxl_canny_controlnet_inpaint(
        strength=1.0,
        prompt=step_2_prompt,
        image=image, # Original image
        mask_image=mask, # Orignial Mask
        control_image=first_layer_canny_edge, # New canny!!
        width=job_input['width'],
        height=job_input['height'],
        negative_prompt=job_input['step_2_negative_prompt'],
        prompt_2=job_input['step_2_prompt_2'],
        negative_prompt_2=job_input['step_2_negative_prompt_2'],
        controlnet_conditioning_scale=job_input['step_2_controlnet_conditioning_scale'],
        num_inference_steps=job_input['step_2_num_inference_steps'],
        guidance_scale=job_input['step_2_guidance_scale'],
        num_images_per_prompt=job_input['step_2_num_images'],
        generator=generator,
    ).images

    return second_step_outpaint


@torch.inference_mode()
def generate_image(job):
    '''
    Generate an image from text using your Model
    '''
    job_input = job["input"]

    # Input validation
    validated_input = validate(job_input, INPUT_SCHEMA)

    if 'errors' in validated_input:
        return {"error": validated_input['errors']}
    job_input = validated_input['validated_input']

    #inpaint_output = sdxl_inpaint_jobs(job_input)
    inpaint_output = sdxl_multi_step_inpaint(job_input)
    image_urls = _save_and_upload_images(inpaint_output, job['id'])

    # Maybe some quick refining here to do some very slight touch oups
    #TODO: but that will come later.

    results = {
        "images": image_urls,
        "image_url": image_urls[0],
        "seed": job_input['seed']
    }

    return results

runpod.serverless.start({"handler": generate_image})
