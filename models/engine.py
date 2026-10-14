import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import pandas as pd
import faiss
import numpy as np
import os
import zlib

COLUMNS = ["image_path", "color", "category", "price", "rating"]
BASE_PRICE = {"shoe": 50, "watch": 150, "laptop": 800, "backpack": 40, "shirt": 25}


def _normalize(v):
    """L2-normalise rows so embeddings can be compared/blended on equal footing."""
    v = np.asarray(v, dtype="float32")
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-8)

class OmniSearchEngine:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.index = None
        self.metadata = None
        
        # Pre-cache common labels for zero-shot (Feature: Speed Optimization)
        self.cat_labels = ["shoe", "watch", "laptop", "backpack", "shirt"]
        self.color_labels = ["red", "blue", "green", "black", "white"]
        self._label_embs = None

    def _ensure_tensor(self, output):
        """Helper to extract tensor from transformer output objects if necessary."""
        if isinstance(output, torch.Tensor):
            return output
        if hasattr(output, "pooler_output"):
            return output.pooler_output
        if hasattr(output, "last_hidden_state"):
            return output.last_hidden_state[:, 0, :] # Use CLS token
        # For CLIP specifically
        if hasattr(output, "image_embeds"):
            return output.image_embeds
        if hasattr(output, "text_embeds"):
            return output.text_embeds
        return output # Fallback

    def get_text_embedding(self, text):
        inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model.get_text_features(**inputs)
        return self._ensure_tensor(outputs).cpu().numpy()

    def get_image_embedding(self, image_path):
        if isinstance(image_path, str):
            image = Image.open(image_path).convert("RGB")
        else:
            image = image_path.convert("RGB")
            
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)
        return self._ensure_tensor(outputs).cpu().numpy()

    def _label_embeddings(self):
        """Normalised zero-shot label embeddings, computed once and cached."""
        if self._label_embs is None:
            cats = np.vstack([self.get_text_embedding(f"a photo of a {l}") for l in self.cat_labels])
            colors = np.vstack([self.get_text_embedding(f"a {l} colored object") for l in self.color_labels])
            self._label_embs = (_normalize(cats), _normalize(colors))
        return self._label_embs

    def _tag(self, emb):
        """Zero-shot (category, color) for one image embedding."""
        cat_norm, color_norm = self._label_embeddings()
        img = _normalize(emb)
        return (self.cat_labels[int(np.argmax(img @ cat_norm.T))],
                self.color_labels[int(np.argmax(img @ color_norm.T))])

    @staticmethod
    def _simulated_price_rating(category, key):
        """Demo price/rating. Seeded from the image path so a product keeps the same values across
        restarts and re-indexing (unseeded randoms made prices change every time the app started)."""
        rng = np.random.default_rng(zlib.crc32(str(key).encode()))
        price = BASE_PRICE.get(category, 50) + int(rng.integers(-10, 50))
        rating = round(float(rng.uniform(3.5, 5.0)), 1)
        return price, rating

    def build_index(self, image_paths):
        """Builds a FAISS index and stores metadata. Uses zero-shot classification for auto-tagging."""
        embeddings, rows = [], []

        print(f"Indexing {len(image_paths)} images with auto-tagging...")
        for path in image_paths:
            try:
                emb = self.get_image_embedding(path)
                category, color = self._tag(emb)
                price, rating = self._simulated_price_rating(category, path)
            except Exception as e:
                print(f"Error processing {path}: {e}")
                continue
            embeddings.append(emb)
            rows.append({"image_path": path, "color": color, "category": category, "price": price, "rating": rating})

        if not embeddings:
            raise ValueError("No images were successfully indexed.")

        embeddings = np.ascontiguousarray(np.vstack(embeddings), dtype="float32")
        faiss.normalize_L2(embeddings)

        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)
        self.metadata = pd.DataFrame(rows, columns=COLUMNS)

    def add_image(self, image, save_path):
        """Adds a single image to the catalog dynamically."""
        if not isinstance(image, Image.Image):
            image = Image.open(image)
        image = image.convert("RGB")  # uploads can be RGBA/palette/CMYK; embed and save a consistent mode

        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        image.save(save_path)

        emb = self.get_image_embedding(image)
        category, color = self._tag(emb)
        price, rating = self._simulated_price_rating(category, save_path)

        emb_idx = np.ascontiguousarray(emb, dtype="float32")
        faiss.normalize_L2(emb_idx)

        if self.index is None:
            self.index = faiss.IndexFlatIP(emb_idx.shape[1])
            self.metadata = pd.DataFrame(columns=COLUMNS)

        self.index.add(emb_idx)
        new_row = pd.DataFrame([{"image_path": save_path, "color": color, "category": category,
                                 "price": price, "rating": rating}], columns=COLUMNS)
        self.metadata = pd.concat([self.metadata, new_row], ignore_index=True)

        return {"category": category, "color": color, "price": price, "rating": rating}

    def search(self, query, query_type="text", top_k=5, filters=None, text_query=None):
        """
        Supports:
        - Text-only
        - Image-only
        - Multi-modal (Image + Text) (Feature: Image-to-Image with refinement)
        """
        if self.index is None or self.metadata is None or len(self.metadata) == 0:
            return pd.DataFrame(columns=COLUMNS + ["score"])  # empty catalog: nothing to match yet

        if query_type == "text":
            query_emb = self.get_text_embedding(query)
        elif query_type == "image":
            query_emb = self.get_image_embedding(query)
            # Feature: If text_query is provided with an image, we blend embeddings
            if text_query:
                text_emb = self.get_text_embedding(text_query)
                # Weighted blend: 70% Image, 30% Text. Normalise both first: raw CLIP image and text
                # vectors have different lengths, so blending them as-is does not give a true 70/30 mix.
                query_emb = 0.7 * _normalize(query_emb) + 0.3 * _normalize(text_emb)
        else:
            raise ValueError("Invalid query_type")
            
        query_emb = query_emb.astype('float32')
        faiss.normalize_L2(query_emb)
        
        # Search all
        distances, indices = self.index.search(query_emb, len(self.metadata))
        
        results = pd.DataFrame({
            "image_path": self.metadata.iloc[indices[0]]["image_path"].values,
            "color": self.metadata.iloc[indices[0]]["color"].values,
            "category": self.metadata.iloc[indices[0]]["category"].values,
            "price": self.metadata.iloc[indices[0]]["price"].values,
            "rating": self.metadata.iloc[indices[0]]["rating"].values,
            "score": distances[0]
        })

        # Apply Filters (Price, Rating, Color, Category)
        if filters:
            if filters.get("category") and filters["category"] != "All":
                results = results[results["category"] == filters["category"].lower()]
            if filters.get("color") and filters["color"] != "All":
                results = results[results["color"] == filters["color"].lower()]
            if filters.get("max_price") is not None:
                results = results[results["price"] <= filters["max_price"]]
            if filters.get("min_rating") is not None:
                results = results[results["rating"] >= filters["min_rating"]]
                    
        return results.head(top_k)

if __name__ == "__main__":
    print("OmniSearch Engine V2 Ready.")
