"""
app.py

Streamlit UI for the multi-stage local image search engine.
Supports:
  - Folder selection
  - Index building / loading
  - Multi-stage search with CLIP + Grounding DINO + Florence-2
  - Score breakdown display
"""

import os
import streamlit as st
import tkinter as tk
from tkinter import filedialog

st.set_page_config(page_title="AI Image Search", layout="wide")
st.title("🖼️ Local AI Image Search Engine")
st.write("Search images using natural language with object detection, counting, and spatial reasoning.")

INDEX_PATH = "data/faiss_index.bin"
METADATA_PATH = "data/metadata.pkl"


@st.cache_resource
def load_pipeline(use_detector, use_verifier):
    """
    Load the full search pipeline. Cached so models load only once.
    """
    from search_pipeline import SearchPipeline
    pipeline = SearchPipeline(
        use_detector=use_detector,
        use_verifier=use_verifier
    )

    # Auto-load saved index if available
    if os.path.exists(INDEX_PATH) and os.path.exists(METADATA_PATH):
        try:
            pipeline.load_index(INDEX_PATH, METADATA_PATH)
        except Exception as e:
            print(f"Could not auto-load index: {e}")

    return pipeline


def select_folder():
    """Open native folder picker dialog."""
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    folder_selected = filedialog.askdirectory()
    root.destroy()
    return folder_selected


# ---- Sidebar: Model Selection ----
st.sidebar.header("🧠 Model Settings")
use_detector = st.sidebar.checkbox("Enable Grounding DINO (detection + counting)", value=True)
use_verifier = st.sidebar.checkbox("Enable Florence-2 (caption verification)", value=True)

# Load pipeline based on selections
pipeline = load_pipeline(use_detector, use_verifier)

# ---- Sidebar: Index Settings ----
st.sidebar.header("📁 Index Settings")

if "folder_path" not in st.session_state:
    st.session_state.folder_path = "sample_images"

new_folder_path = st.sidebar.text_input(
    "Image Folder Path",
    value=st.session_state.folder_path
)
st.session_state.folder_path = new_folder_path
folder_path = st.session_state.folder_path

if st.sidebar.button("📂 Browse Folder"):
    selected = select_folder()
    if selected:
        st.session_state.folder_path = selected
        st.rerun()

build_button = st.sidebar.button("🔨 Build / Rebuild Index")
reload_button = st.sidebar.button("📥 Reload Saved Index")

# Show index status
if pipeline.search_engine.index is not None:
    st.sidebar.success(f"Index loaded: {len(pipeline.search_engine.image_paths)} images")
else:
    st.sidebar.warning("No index loaded")

st.sidebar.write(f"Folder: `{folder_path}`")

if build_button:
    if os.path.exists(folder_path):
        with st.spinner("Building image index..."):
            total = pipeline.build_index(folder_path)
            pipeline.save_index(INDEX_PATH, METADATA_PATH)
        st.sidebar.success(f"Indexed {total} images!")
    else:
        st.sidebar.error("Folder does not exist.")

if reload_button:
    try:
        pipeline.load_index(INDEX_PATH, METADATA_PATH)
        st.sidebar.success(f"Loaded: {len(pipeline.search_engine.image_paths)} images")
    except Exception as e:
        st.sidebar.error(f"Error: {e}")

# ---- Main: Search ----
st.subheader("🔍 Search")

query = st.text_input(
    "Enter your search prompt:",
    placeholder="e.g. a bird flying in the sky and three trees on the left side"
)

col1, col2, col3 = st.columns(3)
with col1:
    top_k = st.slider("Results", min_value=1, max_value=10, value=5)
with col2:
    candidate_k = st.slider("Candidates (Stage 1)", min_value=5, max_value=50, value=20)
with col3:
    clip_weight = st.slider("CLIP weight", min_value=0.0, max_value=1.0, value=0.4, step=0.05)

detection_weight = 0.3
caption_weight = 0.3

if st.button("🔎 Search", type="primary"):
    if pipeline.search_engine.index is None:
        st.error("Please build or load the index first.")
    elif not query.strip():
        st.warning("Please enter a search prompt.")
    else:
        try:
            with st.spinner("Searching... (this may take a moment for detection + verification)"):
                results, parsed = pipeline.search(
                    query=query,
                    top_k=top_k,
                    candidate_k=candidate_k,
                    clip_weight=clip_weight,
                    detection_weight=detection_weight,
                    caption_weight=caption_weight,
                )

            # Show parsed query
            if parsed["objects"]:
                with st.expander("📋 Parsed Query Conditions", expanded=False):
                    for obj in parsed["objects"]:
                        st.write(f"- **{obj['name']}** | count: {obj['count']} | action: {obj['action']} | location: {obj['location']}")

            if results:
                st.subheader(f"Results ({len(results)} found)")

                for i, result in enumerate(results):
                    image_path = result["image_path"]
                    final_score = result["final_score"]
                    clip_score = result["clip_score"]
                    det_score = result["detection_score"]
                    cap_score = result["caption_score"]

                    with st.container():
                        img_col, info_col = st.columns([1, 1])

                        with img_col:
                            st.image(
                                image_path,
                                caption=f"#{i+1} | Score: {final_score:.4f}",
                                use_container_width=True
                            )
                            st.text(os.path.basename(image_path))

                        with info_col:
                            st.markdown("**Score Breakdown:**")
                            st.write(f"🔵 CLIP Similarity: `{clip_score:.4f}`")
                            st.write(f"🟢 Detection Match: `{det_score:.4f}`")
                            st.write(f"🟡 Caption Match: `{cap_score:.4f}`")
                            st.write(f"⭐ **Final Score: `{final_score:.4f}`**")

                            # Detection details
                            if result["detection_details"]:
                                with st.expander("🔍 Detection Details"):
                                    for detail in result["detection_details"]:
                                        st.write(detail)

                            # Caption details
                            if result["caption"]:
                                with st.expander("📝 Generated Caption"):
                                    st.write(result["caption"])

                            if result["caption_details"]:
                                with st.expander("✅ Caption Verification"):
                                    for detail in result["caption_details"]:
                                        st.write(detail)

                    st.divider()
            else:
                st.info("No matching images found.")

        except Exception as e:
            st.error(f"Search failed: {e}")
            import traceback
            st.code(traceback.format_exc())