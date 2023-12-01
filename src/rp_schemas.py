INPUT_SCHEMA = {
    'image_url': {
        'type': str,
        'required': False,
        'default': None
    },
    'mask_url': {
        'type': str,
        'required': False,
        'default': None
    },
    'height': {
        'type': int,
        'required': False,
        'default': 1024
    },
    'width': {
        'type': int,
        'required': False,
        'default': 1024
    },
    'seed': {
        'type': int,
        'required': False,
        'default': None
    },
    # STEP 1 PARAMS
    'step_1_prompt': {
        'type': str,
        'required': False,
        'default': None
    },
    'step_1_negative_prompt': {
        'type': str,
        'required': False,
        'default': None
    },
    'step_1_prompt_2': {
        'type': str,
        'required': False,
        'default': "pretty, beautiful, realism, high quality, high fidelity, detailed,  raw, furniture, interior, sharp, clear, high resolution"
    },
    'step_1_negative_prompt_2': {
        'type': str,
        'required': False,
        'default': "ugly, deformed, noisy, blurry, distorted, out of focus, bad structure, extra table legs, poorly drawn furniture, poorly framed furniture, bad lighting",
    },
    'step_1_num_inference_steps': {
        'type': int,
        'required': False,
        'default': 25
    },
    'step_1_guidance_scale': {
        'type': float,
        'required': False,
        'default': 7
    },
    'step_1_strength': {
        'type': float,
        'required': False,
        'default': 1.0
    },
    'step_1_num_images': {
        'type': int,
        'required': False,
        'default': 1,
        'constraints': lambda img_count: 3 > img_count > 0
    },
    'step_1_controlnet_conditioning_scale': {
        'type': float,
        'required': False,
        'default': 0.5
    },
    'step_1_refiner_num_inference_steps': {
        'type': int,
        'required': False,
        'default': 30
    },
    # STEP 2 PARAMS
    'step_2_prompt': {
        'type': str,
        'required': False,
        'default': None,
    },
    'step_2_negative_prompt': {
        'type': str,
        'required': False,
        'default':  "ugly, deformed, noisy, blurry, distorted, out of focus, bad structure, extra table legs, poorly drawn furniture, poorly framed furniture, bad lighting"
    },
    'step_2_prompt_2': {
        'type': str,
        'required': False,
        'default': "pretty, beautiful, realism, high quality, high fidelity, detailed,  raw, furniture, interior, sharp, clear, high resolution"
    },
    'step_2_negative_prompt_2': {
        'type': str,
        'required': False,
        'default': None
    },
    'step_2_num_inference_steps': {
        'type': int,
        'required': False,
        'default': 50
    },
    'step_2_guidance_scale': {
        'type': float,
        'required': False,
        'default': 7
    },
    'step_2_strength': {
        'type': float,
        'required': False,
        'default': 1.0
    },
    'step_2_num_images': {
        'type': int,
        'required': False,
        'default': 1,
        'constraints': lambda img_count: 3 > img_count > 0
    },
    'step_2_controlnet_conditioning_scale': {
        'type': float,
        'required': False,
        'default': 0.5
    },
    'step_2_refiner_num_inference_steps': {
        'type': int,
        'required': False,
        'default': 8
    },
}
