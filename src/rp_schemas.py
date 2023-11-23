INPUT_SCHEMA = {
    'model': {
      'type': str,
      'required': True
    },
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
    'prompt': {
        'type': str,
        'required': False,
        'default': None
    },
    'negative_prompt': {
        'type': str,
        'required': False,
        'default': None
    },
    'prompt_2': {
        'type': str,
        'required': False,
        'default': None
    },
    'negative_prompt_2': {
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
    'scheduler': {
        'type': str,
        'required': False,
        'default': 'DDIM'
    },
    'num_inference_steps': {
        'type': int,
        'required': False,
        'default': 25
    },
    'refiner_inference_steps': {
        'type': int,
        'required': False,
        'default': 50
    },
    'guidance_scale': {
        'type': float,
        'required': False,
        'default': 7.5
    },
    'strength': {
        'type': float,
        'required': False,
        'default': 0.3
    },
    'num_images': {
        'type': int,
        'required': False,
        'default': 1,
        'constraints': lambda img_count: 3 > img_count > 0
    }
}
