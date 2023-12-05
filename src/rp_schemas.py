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
        'default': "Do not change the furniture in the scene. Multiple furniture items in the room, chairs in front of the table, strange furniture proportions, bad interior design."
    },
    'step_1_prompt_2': {
        'type': str,
        'required': False,
        'default': None
    },
    'step_1_negative_prompt_2': {
        'type': str,
        'required': False,
        'default': "ugly, messy, cluttered, out of frame, out of focus, multiple products, full room, deformed, blurry, bad proportions, gross proportions, missing legs, extra legs, duplicate, low quality, bad positioning, bad interior design, ugly interior, ugly room, ugly furniture, messy room, messy interior, garbage, dirty, unorganized",
    },
    'step_1_num_inference_steps': {
        'type': int,
        'required': False,
        'default': 50
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
        'default': 0.3,
    },
    'step_1_refiner_num_inference_steps': {
        'type': int,
        'required': False,
        'default': 15,
    },
    'step_1_refiner_strength': {
        'type': float,
        'required': False,
        'default': 0.5
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
        'default': "Floating furniture, strangly positioned furniture, furniture in the wrong place. strange furniture proportions. Bad room proportions."
    },
    'step_2_prompt_2': {
        'type': str,
        'required': False,
        'default': None,
    },
    'step_2_negative_prompt_2': {
        'type': str,
        'required': False,
        'default':"ugly, messy, cluttered, out of frame, out of focus, multiple products, full room, deformed, blurry, bad proportions, gross proportions, missing legs, extra legs, duplicate, low quality, bad positioning, bad interior design, ugly interior, ugly room, ugly furniture, messy room, messy interior, garbage, dirty, unorganized"
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
        'default': 0.4
    },
    'step_2_refiner_num_inference_steps': {
        'type': int,
        'required': False,
        'default': 15,
    },
    'step_2_refiner_strength': {
        'type': float,
        'required': False,
        'default': 0.5
    },
    # STEP 3 (QUICK REFINER)
    'step_3_refiner_num_inference_steps': {
        'type': int,
        'required': False,
        'default': 8,
    },

}
