import os
import pickle
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import faiss
import open_clip


class ImageSearchEngine:
    def __init__(self, model_name="ViT-B-32", pretrained="laion2b_s34b_b79k"):
        """
        Initialize model, tokenizer, and preprocessing pipeline.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)

        self.model.to(self.device)
        self.model.eval()

        self.image_paths = []
        self.index = None
        self.embedding_dim = None

    def is_image_file(self, filename):
        """
        Check whether a file is a valid image.
        """
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        ext = os.path.splitext(filename)[1].lower()
        return ext in valid_extensions

    def get_all_image_paths(self, folder_path):
        """
        Recursively collect all image paths from a folder.
        """
        image_paths = []
        for root, _, files in os.walk(folder_path):
            for file in files:
                if self.is_image_file(file):
                    image_paths.append(os.path.join(root, file))
        return image_paths

    def encode_image(self, image_path):
        """
        Convert one image into a normalized embedding vector.
        """
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image_tensor)
            image_features /= image_features.norm(dim=-1, keepdim=True)

        return image_features.cpu().numpy().astype("float32")

    def encode_text(self, text):
        """
        Convert text prompt into a normalized embedding vector.
        """
        text_tokens = self.tokenizer([text]).to(self.device)

        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)

        return text_features.cpu().numpy().astype("float32")

    def build_index(self, folder_path):
        """
        Build FAISS index for all images in the folder.
        """
        all_paths = self.get_all_image_paths(folder_path)

        if not all_paths:
            raise ValueError("No valid images found in the folder.")

        valid_paths = []
        embeddings = []

        for path in tqdm(all_paths, desc="Indexing images"):
            try:
                embedding = self.encode_image(path)
                embeddings.append(embedding)
                valid_paths.append(path)
            except Exception as e:
                print(f"Skipping {path}: {e}")

        if not embeddings:
            raise ValueError("No embeddings could be created.")

        self.image_paths = valid_paths
        embeddings = np.vstack(embeddings)
        self.embedding_dim = embeddings.shape[1]

        # Since embeddings are normalized, inner product ~ cosine similarity
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(embeddings)

        return len(self.image_paths)

    def search(self, query, top_k=5, fetch_k=20, min_score=0.18, relative_threshold=0.90):
        """
        Search images by text query using semantic similarity.
        Filters weak matches using:
        - minimum absolute score
        - relative threshold from best score
        """
        if self.index is None:
            raise ValueError("Index not built or loaded.")

        query_embedding = self.encode_text(query)
        scores, indices = self.index.search(query_embedding, fetch_k)

        results = []
        top_score = float(scores[0][0]) if len(scores[0]) > 0 else 0.0

        for score, idx in zip(scores[0], indices[0]):
            score = float(score)

            if idx >= len(self.image_paths):
                continue

            if score < min_score:
                continue

            if top_score > 0 and score < top_score * relative_threshold:
                continue

            results.append({
                "image_path": self.image_paths[idx],
                "score": score
            })

        return results[:top_k]

    def save_index(self, index_path="data/faiss_index.bin", metadata_path="data/metadata.pkl"):
        """
        Save FAISS index and metadata to disk.
        """
        if self.index is None:
            raise ValueError("No index to save.")

        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        faiss.write_index(self.index, index_path)

        metadata = {
            "image_paths": self.image_paths,
            "embedding_dim": self.embedding_dim
        }

        with open(metadata_path, "wb") as f:
            pickle.dump(metadata, f)

    def load_index(self, index_path="data/faiss_index.bin", metadata_path="data/metadata.pkl"):
        """
        Load FAISS index and metadata from disk.
        """
        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            raise FileNotFoundError("Saved index or metadata not found.")

        self.index = faiss.read_index(index_path)

        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)

        self.image_paths = metadata["image_paths"]
        self.embedding_dim = metadata["embedding_dim"]