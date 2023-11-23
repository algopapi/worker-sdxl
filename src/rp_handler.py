'''
Contains the handler function that will be called by the serverless.
'''

import base64
import concurrent.futures
import os

import runpod
import torch
from diffusers import (ControlNetModel, DDIMScheduler,
                       DPMSolverMultistepScheduler, EulerDiscreteScheduler,
                       LMSDiscreteScheduler, PNDMScheduler,
                       StableDiffusionXLControlNetInpaintPipeline,
                       StableDiffusionXLImg2ImgPipeline,
                       StableDiffusionXLInpaintPipeline,
                       StableDiffusionXLPipeline)
from diffusers.utils import load_image
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
            torch_dtype=torch.float16, variant="fp16", use_safetensors=True, add_watermarker=False
        ).to("cuda", silence_dtype_warnings=True)
        sdxl_outpaint_pipe.enable_xformers_memory_efficient_attention()
        return sdxl_outpaint_pipe
    
    def load_sdxl_canny_controlnet_inpaint(self):
        canny_controlnet = ControlNetModel.from_pretrained(
            "diffusers/controlnet-canny-sdxl-1.0",
            use_safetensors=True, add_watermarker=False
        )

        sdxl_controlnet_outpaint_pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
            "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
            controlnet=canny_controlnet,
            #torch_dtype=torch.float32, 
            #variant="fp32", 
            use_safetensors=True,
            add_watermarker=False
        ).to("cuda", silence_dtype_warnings=True)

        sdxl_controlnet_outpaint_pipe.enable_xformers_memory_efficient_attention()
        return sdxl_controlnet_outpaint_pipe
    
    def load_models(self):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_sdxl_inpaint = executor.submit(self.load_sdxl_inpaint)
            future_sdxl_inpaint_canny_controlnet = executor.submit(self.load_sdxl_canny_controlnet_inpaint)
            
            self.sdxl_inpaint = future_sdxl_inpaint.result()
            self.sdxl_canny_controlnet_inpaint = future_sdxl_inpaint_canny_controlnet.result()

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


def sdxl_controlnet_inpaint_job(job_input: INPUT_SCHEMA):
    prompt = job_input['prompt']
    image = job_input['image_url']
    mask = job_input['mask_url']
    control = job_input['control_image_url']
    
    if job_input['seed'] is None:
        job_input['seed'] = int.from_bytes(os.urandom(2), "big")

    generator = torch.Generator("cuda").manual_seed(job_input['seed'])
    
    image = load_image(image).convert("RGB")
    mask = load_image(mask).convert("RGB")
    control_image = load_image(control).convert("RGB")

    inpaint_output = MODELS.sdxl_inpaint(
        prompt=prompt,
        image=image,
        mask=mask,
        control_image=control_image,
        num_inference_steps=job_input['num_inference_steps'],
        guidance_scale=job_input['guidance_scale'],
        output_type="image",
        num_images_per_prompt=job_input['num_images'],
        generator=generator
    ).images

    return inpaint_output


def sdxl_inpaint_job(job_input: INPUT_SCHEMA):
    prompt = job_input['prompt']
    image = job_input['image_url']
    mask = job_input['mask_url']
    
    if job_input['seed'] is None:
        job_input['seed'] = int.from_bytes(os.urandom(2), "big")

    generator = torch.Generator("cuda").manual_seed(job_input['seed'])
    
    image = load_image(image).convert("RGB")
    mask = load_image(mask).convert("RGB")

    inpaint_output = MODELS.sdxl_inpaint(
        prompt=prompt,
        image=image,
        mask_image=mask,
        num_inference_steps=job_input['num_inference_steps'],
        guidance_scale=job_input['guidance_scale'],
        output_type="image",
        num_images_per_prompt=job_input['num_images'],
        wdith=job_input['width'],
        height=job_input['height'],
        generator=generator
    ).images

    return inpaint_output


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

    requested_model = job_input['model']
    if requested_model == "sdxl_inpaint":
        inpaint_output = sdxl_inpaint_job(job_input)
    elif requested_model == "sdxl_controlnet_inpaint":
        inpaint_output = sdxl_controlnet_inpaint_job(job_input)
    else:
        inpaint_output = None
    

    image_urls = _save_and_upload_images(inpaint_output, job['id'])

    results = {
        "images": image_urls,
        "image_url": image_urls[0],
        "seed": job_input['seed']
    }

    if image:
        results['refresh_worker'] = True

    return results


runpod.serverless.start({"handler": generate_image})
