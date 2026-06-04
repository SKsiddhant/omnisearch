import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import pandas as pd
import faiss
import numpy as np
import os

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

    def build_index(self, image_paths):
        """Builds a FAISS index and stores metadata. Uses zero-shot classification for auto-tagging."""
        embeddings = []
        valid_paths = []
        colors = []
        categories = []
        prices = []
        ratings = []
        
        # Pre-calculate label embeddings if not done
        cat_embs = np.vstack([self.get_text_embedding(f"a photo of a {l}") for l in self.cat_labels])
        color_embs = np.vstack([self.get_text_embedding(f"a {l} colored object") for l in self.color_labels])
        
        print(f"Indexing {len(image_paths)} images with auto-tagging...")
        for path in image_paths:
            try:
                # 1. Get Image Embedding
                emb = self.get_image_embedding(path)
                embeddings.append(emb)
                valid_paths.append(path)
                
                # 2. Zero-shot Auto-tagging
                img_norm = emb / (np.linalg.norm(emb) + 1e-8)
                cat_norm = cat_embs / (np.linalg.norm(cat_embs, axis=1, keepdims=True) + 1e-8)
                color_norm = color_embs / (np.linalg.norm(color_embs, axis=1, keepdims=True) + 1e-8)
                
                category = self.cat_labels[np.argmax(img_norm @ cat_norm.T)]
                color = self.color_labels[np.argmax(img_norm @ color_norm.T)]

                colors.append(color)
                categories.append(category)
                
                # 3. Feature: Price & Rating Simulation
                # In a real app, this comes from a DB. Here we simulate based on category.
                base_price = {"shoe": 50, "watch": 150, "laptop": 800, "backpack": 40, "shirt": 25}
                prices.append(base_price.get(category, 50) + np.random.randint(-10, 50))
                ratings.append(round(np.random.uniform(3.5, 5.0), 1))
                
            except Exception as e:
                print(f"Error processing {path}: {e}")
        
        if not embeddings:
            raise ValueError("No images were successfully indexed.")

        embeddings = np.vstack(embeddings).astype('float32')
        faiss.normalize_L2(embeddings)
        
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)
        self.metadata = pd.DataFrame({
            "image_path": valid_paths,
            "color": colors,
            "category": categories,
            "price": prices,
            "rating": ratings
        })
        
    def search(self, query, query_type="text", top_k=5, filters=None, text_query=None):
        """
        Supports:
        - Text-only
        - Image-only
        - Multi-modal (Image + Text) (Feature: Image-to-Image with refinement)
        """
        if query_type == "text":
            query_emb = self.get_text_embedding(query)
        elif query_type == "image":
            query_emb = self.get_image_embedding(query)
            # Feature: If text_query is provided with an image, we blend embeddings
            if text_query:
                text_emb = self.get_text_embedding(text_query)
                # Weighted blend: 70% Image, 30% Text for refinement
                query_emb = 0.7 * query_emb + 0.3 * text_emb
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
            if filters.get("max_price"):
                results = results[results["price"] <= filters["max_price"]]
            if filters.get("min_rating"):
                results = results[results["rating"] >= filters["min_rating"]]
                    
        return results.head(top_k)

if __name__ == "__main__":
    print("OmniSearch Engine V2 Ready.")
