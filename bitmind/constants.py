import os
import torch


WANDB_PROJECT = 'bitmind'
WANDB_ENTITY = 'bitmindai'

DATASET_META = {
    "real": [
        {"path": "bitmind/open-images-v7", "create_splits": False},
        {"path": "bitmind/ffhq-256", "create_splits": False},
        {"path": "bitmind/celeb-a-hq", "create_splits": False},
        {"path": "bitmind/MS-COCO-unique-256", "create_splits": False}
    ],
    "fake": [
        {"path": "bitmind/realvis-xl", "create_splits": False},
        {"path": "bitmind/stable-diffusion-xl", "create_splits": False},
    ]
}

FACE_TRAINING_DATASET_META = {
    "real": [
        {"path": "bitmind/celeb-a-hq_training_faces", "create_splits": False},
        {"path": "bitmind/ffhq-256_training_faces", "create_splits": False},
    ],
    "fake": [
        {"path": "bitmind/celeb-a-hq___stable-diffusion-xl-base-1.0___256_training_faces", "create_splits": False},
        {"path": "bitmind/ffhq-256___stable-diffusion-xl-base-1.0_training_faces", "create_splits": False}
    ]
}

VALIDATOR_MODEL_META = {
    "prompt_generators": [
        {
            "model": "Gustavosta/MagicPrompt-Stable-Diffusion",
            "tokenizer": "gpt2",
            "device": -1
        }
    ],
    "diffusers": [
        {
            "path": "stabilityai/stable-diffusion-xl-base-1.0",
            "use_safetensors": True,
            "torch_dtype": torch.float16,
            "variant": "fp16",
            "pipeline": "StableDiffusionXLPipeline"
        },
        {
            "path": "SG161222/RealVisXL_V4.0",
            "use_safetensors": True,
            "torch_dtype": torch.float16,
            "variant": "fp16",
            "pipeline": "StableDiffusionXLPipeline"
        },
        {
            "path": "Corcelio/mobius",
            "use_safetensors": True,
            "torch_dtype": torch.float16,
            "pipeline": "StableDiffusionXLPipeline"
        },
        {
            "path": 'black-forest-labs/FLUX.1-dev',
            "use_safetensors": True,
            "torch_dtype": torch.bfloat16,
            "generate_args": {
                "guidance_scale": 2,
                "num_inference_steps": {"min": 50, "max": 125},
                "generator": torch.Generator("cuda"),
                "height": [512, 768],
                "width": [512, 768]
            },
            "enable_cpu_offload": False,
            "pipeline": "FluxPipeline"
        }
    ]
}

HUGGINGFACE_CACHE_DIR = os.path.expanduser('~/.cache/huggingface')

TARGET_IMAGE_SIZE = (256, 256)

PROMPT_TYPES = ('random', 'annotation', 'none')

PROMPT_GENERATOR_ARGS = {
    m['model']: m for m in VALIDATOR_MODEL_META['prompt_generators']
}

PROMPT_GENERATOR_NAMES = list(PROMPT_GENERATOR_ARGS.keys())

# args for .from_pretrained
DIFFUSER_ARGS = {
    m['path']: {
        k: v for k, v in m.items()
        if k not in ('path', 'pipeline', 'generate_args', 'enable_cpu_offload')
    } for m in VALIDATOR_MODEL_META['diffusers']
}

GENERATE_ARGS = {
    m['path']: m['generate_args']
    for m in VALIDATOR_MODEL_META['diffusers']
    if 'generate_args' in m
}

DIFFUSER_CPU_OFFLOAD_ENABLED = {
    m['path']: m.get('enable_cpu_offload', False)
    for m in VALIDATOR_MODEL_META['diffusers']
}

DIFFUSER_PIPELINE = {
    m['path']: m['pipeline'] for m in VALIDATOR_MODEL_META['diffusers'] if 'pipeline' in m
}

DIFFUSER_NAMES = list(DIFFUSER_ARGS.keys())

IMAGE_ANNOTATION_MODEL = "Salesforce/blip2-opt-6.7b-coco"

TEXT_MODERATION_MODEL = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit" 
# "meta-llama/Meta-Llama-3.1-8B-Instruct"
