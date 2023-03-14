import torch
from torch.nn.functional import normalize
import clip
import numpy as np


class CLIP:
    
    def __init__(self, model_name: str, device: str = 'cuda'):
        self.model_name = model_name
        self.device = device

        # Initialize model
        self.model, self.preprocess = clip.load(self.model_name, device=self.device)


    def __normalize(self, feature):
        """
        Normalizes a tensor.
        """
        normed_feature = normalize(feature, p=2, dim=1)
        return normed_feature

    
    def encode_image(self, images, norm: bool = True):
        """
        Encodes an image from a tensor using CLIP model.
        """
        with torch.no_grad():
            image_features = self.model.encode_image(images)
            if norm == True:
                # Normalize image encoded features
                image_features = self.__normalize(image_features)
        return image_features


    def encode_text(self, text: str, norm: bool = True):
        """
        Encodes text into a vector using CLIP model.
        NOTE: Encode only one sentence.
        """
        with torch.no_grad():
            tokenized_text = clip.tokenize(text).to(self.device)
            text_features = self.model.encode_text(tokenized_text)
            if norm == True:
                # Normalize text encoded features
                text_features = self.__normalize(text_features)
        return text_features