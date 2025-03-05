from sentence_transformers import SentenceTransformer, util
from typing import List, Union
import numpy as np

class TextSimilarityModel:
    """
    A class to handle text embedding and semantic similarity computation using SentenceTransformers.
    """
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the class and load the default model.
        """
        self.model_name = model_name
        self.model = None
        self.load_model()
    
    def load_model(self):
        """
        Loads the SentenceTransformer model.
        """
        self.model = SentenceTransformer(self.model_name)
    
    def fetch_vectors(self, texts: List[str]) -> np.ndarray:
        """
        Converts a list of texts into their corresponding embeddings (vectors).
        
        Args:
            texts (List[str]): A list of text strings to be encoded.
            
        Returns:
            np.ndarray: A 2D array of embeddings (vectors) for the input texts.
        """
        if self.model is None:
            raise ValueError("Model is not loaded. Please call `load_model()` to load the model.")
        return self.model.encode(texts)
    
    def compute_semantic_similarity(self, prompt_vector: np.ndarray, img_descs_vectors: np.ndarray) -> List[float]:
        """
        Computes cosine similarity between the prompt vector and each image description vector.
        """
        if self.model is None:
            raise ValueError("Model is not loaded. Please call `load_model()` to load the model.")
        print(len(prompt_vector.shape), len(img_descs_vectors.shape))
        # Ensure prompt_vector is 2D
        if len(prompt_vector.shape) == 1:
            prompt_vector = np.expand_dims(prompt_vector, axis=0)  # Convert to shape [1, embedding_dim]
        
        # Ensure img_descs_vectors is 2D
        if len(img_descs_vectors.shape) == 1:
            img_descs_vectors = np.expand_dims(img_descs_vectors, axis=0)
    
        # Compute cosine similarities
        similarities = []
        for img_desc_vector in img_descs_vectors:
            if len(img_desc_vector.shape) == 1:
                img_desc_vector = np.expand_dims(img_desc_vector, axis=0)  # Ensure 2D for each vector
            similarity = util.cos_sim(prompt_vector, img_desc_vector).item()  # Convert to scalar
            similarities.append(similarity)
        return similarities