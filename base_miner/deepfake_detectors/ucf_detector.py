import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Ignore INFO and WARN messages

import random
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import yaml
from PIL import Image
from huggingface_hub import hf_hub_download
import gc

from base_miner.UCF.config.constants import (
    CONFIG_PATH,
    WEIGHTS_PATH,
    WEIGHTS_HF_PATH,
    DFB_CKPT,
    BACKBONE_CKPT)

from base_miner.gating_mechanisms import FaceGate

from base_miner.UCF.detectors import DETECTOR
from base_miner.deepfake_detectors import DeepfakeDetector
from base_miner import DETECTOR_REGISTRY, GATE_REGISTRY

@DETECTOR_REGISTRY.register_module(module_name='UCF')
class UCFDetector(DeepfakeDetector):
    """
    This class initializes a pretrained UCF model by loading the necessary configurations, checkpoints,
    and  face detection tools required for training and inference. It sets up the device (CPU or GPU), 
    loads the specified weights, and initializes the face detector and predictor from dlib.

    Attributes:
        config_path (str): Path to the configuration YAML file for the UCF model.
        weights_dir (str): Directory path where UCF and Xception backbone weights are stored.
        weights_hf_repo_name (str): Name of the Hugging Face repository containing the model weights.
        ucf_checkpoint_name (str): Filename of the UCF model checkpoint.
        backbone_checkpoint_name (str): Filename of the backbone model checkpoint.
        predictor_path (str): Path to the dlib face predictor file.
        specific_task_number (int): Number of different fake training dataset/forgery methods
                                    for UCF to disentangle (DeepfakeBench default is 5, the
                                    num of datasets of FF++)."""
    
    def __init__(self, model_name: str = 'UCF', config_path=CONFIG_PATH, weights_dir=WEIGHTS_PATH, 
             weights_hf_repo_name=WEIGHTS_HF_PATH, ucf_checkpoint_name=DFB_CKPT, 
             backbone_checkpoint_name=BACKBONE_CKPT, specific_task_number=5, gate=None):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config_path = Path(config_path)
        self.weights_dir = Path(weights_dir)
        self.hugging_face_repo_name = weights_hf_repo_name
        self.ucf_checkpoint_name = ucf_checkpoint_name
        self.backbone_checkpoint_name = backbone_checkpoint_name
        self.config = self.load_config()
        self.config['specific_task_number'] = specific_task_number
        
        self.init_cudnn()
        self.init_seed()
        self.ensure_weights_are_available(self.ucf_checkpoint_name)
        self.ensure_weights_are_available(self.backbone_checkpoint_name)
        self.gate = GATE_REGISTRY[gate]() if gate else None
        super().__init__(model_name)

    def ensure_weights_are_available(self, model_filename):
        destination_path = self.weights_dir / model_filename
        if not destination_path.parent.exists():
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"Created directory {destination_path.parent}.")
        if not destination_path.exists():
            model_path = hf_hub_download(self.hugging_face_repo_name, model_filename)
            model = torch.load(model_path, map_location=self.device)
            torch.save(model, destination_path)
            print(f"Downloaded {model_filename} to {destination_path}.")
        else:
            print(f"{model_filename} already present at {destination_path}.")

    def load_config(self):
        if not self.config_path.exists():
            raise FileNotFoundError(f"The config file at {self.config_path} does not exist.")
        with self.config_path.open('r') as file:
            return yaml.safe_load(file)
        
    def init_cudnn(self):
        if self.config.get('cudnn'):
            cudnn.benchmark = True

    def init_seed(self):
        seed_value = self.config.get('manualSeed')
        if seed_value:
            random.seed(seed_value)
            torch.manual_seed(seed_value)
            torch.cuda.manual_seed_all(seed_value)

    def load_model(self):
        model_class = DETECTOR[self.config['model_name']]
        self.model = model_class(self.config).to(self.device)
        self.model.eval()
        weights_path = self.weights_dir / self.ucf_checkpoint_name
        try:
            checkpoint = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(checkpoint, strict=True)
            print('Loaded checkpoint successfully.')
        except FileNotFoundError:
            print('Failed to load the pretrained weights.')
    
    def preprocess(self, image, res=256):
        """Preprocess the image for model inference.
        
        Returns:
            torch.Tensor: The preprocessed image tensor, ready for model inference.
        """
        if self.gate: image = self.gate(image, res)
        # Convert image to RGB format to ensure consistent color handling.
        image = image.convert('RGB')
    
        # Define transformation sequence for image preprocessing.
        transform = transforms.Compose([
            transforms.Resize((res, res), interpolation=Image.LANCZOS),  # Resize image to specified resolution.
            transforms.ToTensor(),  # Convert the image to a PyTorch tensor.
            transforms.Normalize(mean=self.config['mean'], std=self.config['std'])  # Normalize the image tensor.
        ])
        
        # Apply transformations and add a batch dimension for model inference.
        image_tensor = transform(image).unsqueeze(0)
        
        # Move the image tensor to the specified device (e.g., GPU).
        return image_tensor.to(self.device)

    def infer(self, image_tensor):
        """ Perform inference using the model. """
        with torch.no_grad():
            self.model({'image': image_tensor}, inference=True)
        return self.model.prob[-1]

    def __call__(self, image: Image) -> float:
        image_tensor = self.preprocess(image)
        return self.infer(image_tensor)
    
    def free_memory(self):
        """ Frees up memory by setting model and large data structures to None. """
        print("Freeing up memory...")

        if self.model is not None:
            self.model.cpu()  # Move model to CPU to free up GPU memory (if applicable)
            del self.model
            self.model = None

        if self.face_detector is not None:
            del self.face_detector
            self.face_detector = None

        if self.face_predictor is not None:
            del self.face_predictor
            self.face_predictor = None

        gc.collect()

        # If using GPUs and PyTorch, clear the cache as well
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("Memory freed successfully.")