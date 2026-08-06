import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
from typing import Optional, Tuple, Dict, Any
import os

class DinoFeatureExtractor:
    """
    Extracts semantic features using DINOv2 to enable texture-aware clustering.
    """
    def __init__(self, model_type: str = 'dinov2_vits14'):
        print(f"Loading DINOv2 model ({model_type})...")
        # Using torch.hub to load the model as it's the easiest integration
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = torch.hub.load('facebookresearch/dinov2', model_type).to(self.device)
        self.model.eval()
        
        self.transform = T.Compose([
            T.Resize(518, interpolation=T.InterpolationMode.BICUBIC), # DINOv2 likes multiples of 14
            T.CenterCrop(518),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def get_features(self, image_path: str) -> np.ndarray:
        img = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        # Get the global feature vector (CLS token)
        features = self.model(input_tensor)
        return features.cpu().numpy().flatten()

    @torch.no_grad()
    def get_patch_features(self, image_path: str) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Extracts features for patches (spatial tokens) to allow pixel-like clustering.
        """
        img = Image.open(image_path).convert('RGB')
        w, h = img.size
        # Resize to be divisible by 14 (DINOv2 patch size)
        w_new, h_new = (w // 14) * 14, (h // 14) * 14
        img_resized = img.resize((w_new, h_new), Image.LANCZOS)
        
        input_tensor = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])(img_resized).unsqueeze(0).to(self.device)
        
        # DINOv2 forward_features returns cls_token and patch_tokens
        out = self.model.get_intermediate_layers(input_tensor, n=1)[0]
        
        # out shape is [1, num_patches, embedding_dim]
        # num_patches = (w_new/14) * (h_new/14)
        patch_features = out.cpu().numpy()[0]
        grid_shape = (h_new // 14, w_new // 14)
        
        return patch_features, grid_shape

class DinoClassifier:
    """
    Clusters image regions based on DINOv2 features.
    """
    def __init__(self, n_clusters: int = 5):
        self.extractor = DinoFeatureExtractor()
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)

    def analyze_image(self, image_path: str):
        print(f"Analyzing textures/semantics in {image_path}...")
        features, grid_shape = self.extractor.get_patch_features(image_path)
        
        # Cluster the patches
        labels = self.kmeans.fit_predict(features)
        
        # Reshape labels back to grid
        label_grid = labels.reshape(grid_shape)
        return label_grid, features

if __name__ == "__main__":
    # Quick test logic
    test_img = "FLACA/mural_flat.png"
    if os.path.exists(test_img):
        classifier = DinoClassifier(n_clusters=4)
        labels, _ = classifier.analyze_image(test_img)
        print(f"Analysis complete. Resulting label grid shape: {labels.shape}")
        
        # To visualize, we'd upscale this grid to the original image size
        # and overlay it as we do in ColorClassifier_manual.py
    else:
        print(f"Test image {test_img} not found.")
