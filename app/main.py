import streamlit as st
from PIL import Image
import os
import sys

# Add parent directory to path to import engine
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.engine import OmniSearchEngine

st.set_page_config(page_title="OmniSearch V2 | Amazon ML SS", layout="wide")

# Custom CSS for a more "Amazon" feel
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { background-color: #ff9900; color: white; border-radius: 8px; width: 100%; }
    .product-card { border: 1px solid #ddd; padding: 10px; border-radius: 8px; background: white; margin-bottom: 20px; }
    .price { color: #B12704; font-size: 1.2em; font-weight: bold; }
    .rating { color: #ffa41c; }
    </style>
    """, unsafe_allow_html=True)

st.title("🛍️ OmniSearch V2: Next-Gen Discovery")
st.markdown("Implemented for **Amazon ML Summer School 2026**. Featuring Multimodal RAG and Zero-Shot Tagging.")

@st.cache_resource
def load_engine():
    engine = OmniSearchEngine()
    image_dir = "data/sample_images"
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if image_paths:
        with st.spinner("Indexing catalog with zero-shot auto-tagging..."):
            engine.build_index(image_paths)
    else:
        # Initialize empty index if no images
        engine.metadata = None 
    return engine

engine = load_engine()

# --- Sidebar: Add New Product ---
with st.sidebar:
    st.divider()
    with st.expander("➕ Add New Product to Catalog"):
        new_img_file = st.file_uploader("Upload Product Image", type=["jpg", "png", "jpeg"], key="admin_upload")
        if new_img_file:
            st.image(new_img_file, caption="Preview", use_column_width=True)
            if st.button("Add to Catalog"):
                with st.spinner("Saving and indexing..."):
                    # Save to data/sample_images
                    img_name = f"custom_{len(os.listdir('data/sample_images')) + 1}.png"
                    save_path = os.path.join("data/sample_images", img_name)
                    
                    try:
                        info = engine.add_image(Image.open(new_img_file), save_path)
                        st.success(f"Added! Detected: {info['color']} {info['category']}")
                        st.balloons()
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error adding image: {e}")

if not engine or (engine.metadata is None and not os.listdir("data/sample_images")):
    st.error("No images found in `data/sample_images`. Use the sidebar to add some!")
    # We don't stop here anymore, we allow adding via sidebar

col1, col2 = st.columns([1, 3])

with col1:
    st.subheader("🔍 Discovery Mode")
    search_type = st.radio("Primary Input:", ["Natural Language", "Visual Reference"])
    
    query = None
    refinement = None
    
    if search_type == "Natural Language":
        query = st.text_input("What are you looking for?", placeholder="e.g., green waterproof backpack")
    else:
        query = st.file_uploader("Upload product photo", type=["jpg", "png", "jpeg"])
        refinement = st.text_input("Refine with text (Optional):", placeholder="e.g., but in red")

    st.divider()
    st.subheader("⚙️ Smart Filters")
    
    cat_filter = st.selectbox("Category:", ["All", "Shoe", "Watch", "Laptop", "Backpack", "Shirt"])
    color_filter = st.selectbox("Color:", ["All", "Red", "Blue", "Green", "Black", "White"])
    
    max_price = st.slider("Max Price ($):", 10, 1000, 1000)
    min_rating = st.slider("Min Rating (Stars):", 0.0, 5.0, 0.0, step=0.5)
    
    top_k = st.number_input("Results count:", 1, 10, 6)
    
    search_button = st.button("Search Catalog")

with col2:
    if search_button:
        if query:
            with st.spinner("AI is analyzing your request..."):
                try:
                    filters = {
                        "category": cat_filter,
                        "color": color_filter,
                        "max_price": max_price,
                        "min_rating": min_rating
                    }
                    
                    if search_type == "Visual Reference":
                        # Convert uploaded file to PIL Image
                        query_img = Image.open(query)
                        results = engine.search(query_img, query_type="image", top_k=top_k, filters=filters, text_query=refinement)
                    else:
                        results = engine.search(query, query_type="text", top_k=top_k, filters=filters)
                    
                    if results.empty:
                        st.warning("No products match your criteria. Try loosening the filters.")
                    else:
                        st.write(f"Found {len(results)} relevant items.")
                        # Display results in a grid
                        rows = (len(results) + 2) // 3
                        for r in range(rows):
                            cols = st.columns(3)
                            for c in range(3):
                                idx = r * 3 + c
                                if idx < len(results):
                                    row = results.iloc[idx]
                                    with cols[c]:
                                        with st.container():
                                            st.markdown('<div class="product-card">', unsafe_allow_html=True)
                                            st.image(Image.open(row['image_path']), use_column_width=True)
                                            st.markdown(f"**{row['color'].title()} {row['category'].title()}**")
                                            st.markdown(f'<span class="price">${row["price"]}</span>', unsafe_allow_html=True)
                                            st.markdown(f'<span class="rating">{"★" * int(row["rating"])}{"☆" * (5-int(row["rating"]))} ({row["rating"]})</span>', unsafe_allow_html=True)
                                            st.caption(f"Relevance Score: {row['score']:.2f}")
                                            st.markdown('</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Discovery Error: {e}")
        else:
            st.info("Please provide an input to start discovery.")
    else:
        # Default view: Show trending (random)
        st.subheader("Trending Now")
        if engine.metadata is not None and not engine.metadata.empty:
            results = engine.metadata.sample(min(6, len(engine.metadata)))
            cols = st.columns(3)
            for i, (idx, row) in enumerate(results.iterrows()):
                with cols[i % 3]:
                    st.image(Image.open(row['image_path']), caption=f"{row['color'].title()} {row['category'].title()}", use_column_width=True)
        else:
            st.info("Catalog is empty. Add products in the sidebar or run the sample generator.")

st.sidebar.markdown(f"""
### 🧬 OmniSearch Tech Stack
- **Vision-Language Model:** CLIP
- **Vector Engine:** FAISS (Approximate Nearest Neighbors)
- **Tagging:** Zero-Shot Inference
- **Backend:** PyTorch + Pandas
- **Device:** `{engine.device.upper()}`
""")
