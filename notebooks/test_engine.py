import sys
import os
import torch
import numpy as np

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.engine import OmniSearchEngine

def test_engine():
    print("Testing OmniSearch Engine...")
    engine = OmniSearchEngine()
    
    image_dir = "data/sample_images"
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_paths:
        print("No images found for testing.")
        return False
        
    engine.build_index(image_paths)
    print(f"Index built with {len(image_paths)} images.")
    
    # Test Text Search
    query = "black shoe"
    print(f"Searching for: '{query}'")
    results = engine.search(query, query_type="text", top_k=3)
    print("Results:")
    print(results)
    
    if len(results) > 0:
        print("Text search test passed.")
    else:
        print("Text search test failed.")
        return False

    return True

if __name__ == "__main__":
    success = test_engine()
    if not success:
        sys.exit(1)
