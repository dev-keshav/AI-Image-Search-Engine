import os
import streamlit as st
from search_engine import ImageSearchEngine

st.set_page_config(page_title="Local Image Search", layout="wide")

st.title("🖼️ Local AI Image Search Engine")
st.write("Search images in your local folder using natural language prompts.")

INDEX_PATH = "data/faiss_index.bin"
METADATA_PATH = "data/metadata.pkl"

@st.cache_resource
def load_engine():
    engine = ImageSearchEngine()

    # Auto-load saved index if present
    if os.path.exists(INDEX_PATH) and os.path.exists(METADATA_PATH):
        try:
            engine.load_index(INDEX_PATH, METADATA_PATH)
        except Exception as e:
            print(f"Could not auto-load index: {e}")

    return engine

engine = load_engine()

st.sidebar.header("Index Settings")
folder_path = st.sidebar.text_input("Image Folder Path", value="sample_images")
build_button = st.sidebar.button("Build / Rebuild Index")
reload_button = st.sidebar.button("Reload Saved Index")

# Show current status
if engine.index is not None:
    st.sidebar.success(f"Index loaded with {len(engine.image_paths)} images.")
else:
    st.sidebar.warning("No index loaded.")

if build_button:
    if os.path.exists(folder_path):
        with st.spinner("Building image index..."):
            engine.build_index(folder_path)
            engine.save_index(INDEX_PATH, METADATA_PATH)
            engine.load_index(INDEX_PATH, METADATA_PATH)
        st.sidebar.success(f"Indexed {len(engine.image_paths)} images successfully.")
    else:
        st.sidebar.error("Folder path does not exist.")

if reload_button:
    try:
        engine.load_index(INDEX_PATH, METADATA_PATH)
        st.sidebar.success(f"Reloaded index with {len(engine.image_paths)} images.")
    except Exception as e:
        st.sidebar.error(f"Error loading saved index: {e}")

query = st.text_input(
    "Enter your search prompt:",
    placeholder="e.g. a white dog running on grass"
)
top_k = st.slider("Number of results", min_value=1, max_value=10, value=5)

if st.button("Search"):
    if engine.index is None:
        st.error("Please build or load the index first.")
    elif not query.strip():
        st.warning("Please enter a search prompt.")
    else:
        try:
            results = engine.search(query, top_k=top_k)

            if results:
                st.subheader("Search Results")

                cols = st.columns(min(5, len(results)))
                for i, result in enumerate(results):
                    image_path = result["image_path"]
                    score = result["score"]

                    with cols[i % len(cols)]:
                        st.image(image_path, caption=f"Score: {score:.4f}", use_container_width=True)
                        st.text(os.path.basename(image_path))
            else:
                st.info("No matching images found.")

        except Exception as e:
            st.error(f"Search failed: {e}")