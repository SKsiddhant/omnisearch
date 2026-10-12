# 🛍️ OmniSearch: Multimodal Discovery Engine

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.24%2B-ff4b4b.svg)](https://streamlit.io/)
[![FAISS](https://img.shields.io/badge/FAISS-Vector_DB-00a3e0.svg)](https://github.com/facebookresearch/faiss)

OmniSearch is a state-of-the-art semantic discovery engine developed for the **Amazon ML Summer School 2026** application. It bridges the gap between textual intent and visual representation using Multimodal Large Language Models (LLMs) to revolutionize how users discover products in a digital catalog.

---

## 🌟 Key Features

- **🔍 Multimodal Search:** Search using natural language ("rugged black backpack") or by uploading an image.
- **⚡ Hybrid Refinement:** Upload an image and refine your search with text (e.g., "this shoe but in red").
- **🏷️ Zero-Shot Auto-Tagging:** Automatically categorizes and extracts attributes (color, category) from images without retraining, powered by CLIP's zero-shot capabilities.
- **📊 Smart Analytics Dashboard:** Interactive UI with real-time filtering by price, ratings, and semantic similarity scores.
- **🚀 High-Performance Indexing:** Uses **FAISS (Facebook AI Similarity Search)** for fast similarity search over the catalog embeddings.

---

## 🏗️ Technical Architecture

OmniSearch utilizes a **Retrieval-Augmented Generation (RAG)** inspired pipeline for visual discovery:

1.  **Embedding Layer:** Uses `OpenAI CLIP (ViT-B/32)` to map both images and text into a shared 512-dimensional latent space.
2.  **Vector Store:** High-dimensional embeddings are indexed using **FAISS IndexFlatIP** with L2 normalization for cosine similarity search.
3.  **Inference Engine:** 
    *   **Text-to-Image:** Computes text embeddings and performs exact nearest-neighbour search.
    *   **Image-to-Image:** Computes query image embeddings for visual similarity.
    *   **Zero-Shot Tagging:** Performs cross-modal classification by comparing image embeddings against predefined label embeddings (Shoe, Watch, Laptop, etc.).
4.  **UI/UX:** Built with **Streamlit**, providing a responsive and intuitive e-commerce discovery interface.

---

## 🛠️ Installation & Usage

### 1. Clone the repository
```bash
git clone https://github.com/SKsiddhant/omnisearch.git
cd omnisearch
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up Environment (Optional but recommended)
```bash
# Set your Hugging Face token for faster model downloads
export HF_TOKEN="your_hugging_face_token"
```

### 4. Generate the sample catalog and run the demo
```bash
python data/generate_samples.py
streamlit run app/main.py
```

---

## 🧪 Tests

```bash
pip install -r requirements-dev.txt
pytest
```
The tests replace CLIP with a tiny fake model, so they run offline in under a second and check the engine's indexing, tagging, filtering and blending logic, not CLIP's accuracy.

---

## 📈 Evaluation

No retrieval benchmark has been run on this project yet, so no accuracy or latency numbers are claimed here.
Prices and ratings shown in the demo are **simulated** (derived deterministically from each image's file name), not real catalog data.
The category and colour tags come from CLIP zero-shot matching against a small fixed label list and can be wrong.

---

## 🎯 Amazon ML Summer School Relevance

This project aligns with several modules of the **Amazon ML Summer School 2026** curriculum:
- **Module 2 (Deep Neural Networks):** Implementation of Transformer-based Vision models.
- **Module 8 (Generative AI & LLMs):** Utilizing foundation models (CLIP) for zero-shot tasks and multimodal embedding spaces.
- **Amazon Leadership Principles:** Directly addresses **Customer Obsession** by creating a more intuitive and powerful search experience for shoppers.

---

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
*Created with ❤️ for the Amazon ML Summer School Application.*
