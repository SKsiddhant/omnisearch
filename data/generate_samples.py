import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image

def create_sample_dataset(output_dir="data/sample_images"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Define some categories and colors to create distinct images
    categories = ["Shoe", "Watch", "Laptop", "Backpack", "Shirt"]
    colors = ["red", "blue", "green", "black", "white"]
    
    print(f"Generating {len(categories) * len(colors)} synthetic images...")
    
    for cat in categories:
        for color in colors:
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.set_facecolor(color)
            ax.text(0.5, 0.5, f"{color} {cat}", 
                    color="gray" if color in ["white", "yellow"] else "white",
                    fontsize=20, ha='center', va='center')
            ax.set_xticks([])
            ax.set_yticks([])
            
            filename = f"{color}_{cat.lower()}.png"
            plt.savefig(os.path.join(output_dir, filename))
            plt.close()

if __name__ == "__main__":
    create_sample_dataset()
    print("Sample dataset created successfully.")
