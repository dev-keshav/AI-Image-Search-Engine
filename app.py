"""
app.py

Streamlit UI for the local multi-stage image search engine.
"""

import os
import time
import traceback
from html import escape
from pathlib import Path

import streamlit as st
import tkinter as tk
from tkinter import filedialog

st.set_page_config(
    page_title="AI Image Search",
    layout="wide",
    initial_sidebar_state="expanded",
)

INDEX_PATH = "data/faiss_index.bin"
METADATA_PATH = "data/metadata.pkl"
STYLE_PATH = Path(__file__).with_name("styles.css")
SUGGESTED_QUERIES = [
    "a red car parked near a building",
    "two birds in the sky",
    "a person standing on the left side",
    "a dog running on grass",
]


def apply_custom_style():
    """Load the app stylesheet from disk."""
    css = STYLE_PATH.read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


apply_custom_style()


def ensure_state():
    """Initialize session state used across reruns."""
    defaults = {
        "folder_path": "sample_images" if os.path.exists("sample_images") else ".",
        "query_input": "",
        "search_results": None,
        "parsed_query": None,
        "search_error": None,
        "search_traceback": None,
        "last_query": "",
        "last_search_seconds": None,
        "collection_notice": None,
        "pipeline_flags": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def clear_search_state():
    """Clear persisted search output when the collection changes."""
    st.session_state.search_results = None
    st.session_state.parsed_query = None
    st.session_state.search_error = None
    st.session_state.search_traceback = None
    st.session_state.last_query = ""
    st.session_state.last_search_seconds = None


@st.cache_resource
def load_pipeline(use_detector, use_verifier):
    """
    Load the full search pipeline. Cached so models load only once.
    """
    from search_pipeline import SearchPipeline

    pipeline = SearchPipeline(
        use_detector=use_detector,
        use_verifier=use_verifier,
    )

    if os.path.exists(INDEX_PATH) and os.path.exists(METADATA_PATH):
        try:
            pipeline.load_index(INDEX_PATH, METADATA_PATH)
        except Exception as exc:
            print(f"Could not auto-load index: {exc}")

    return pipeline


def run_with_loader(message, callback, *args, **kwargs):
    """Show the full-screen loader around long operations and return duration."""
    start = time.perf_counter()
    with st.spinner(message):
        result = callback(*args, **kwargs)
    return result, time.perf_counter() - start


def select_folder():
    """Open native folder picker dialog."""
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    folder_selected = filedialog.askdirectory()
    root.destroy()
    return folder_selected


def format_stage_state(enabled):
    return "Enabled" if enabled else "Disabled"


def format_query_conditions(parsed_query):
    """Create compact human-readable labels from parsed query objects."""
    chips = []
    if not parsed_query:
        return chips

    for obj in parsed_query.get("objects", []):
        parts = [obj.get("name", "object")]
        count = obj.get("count")
        action = obj.get("action")
        location = obj.get("location")

        if count not in (None, "", 1):
            parts.append(f"x{count}")
        elif count == 1:
            parts.append("single")

        if action:
            parts.append(action)
        if location:
            parts.append(location)

        chips.append(" | ".join(parts))

    return chips


def render_score_meter(label, value, tone):
    """Return HTML for a score bar."""
    safe_value = max(0.0, min(float(value), 1.0))
    return f"""
    <div class="score-row">
        <div class="score-line">
            <span>{escape(label)}</span>
            <strong>{safe_value:.3f}</strong>
        </div>
        <div class="score-track">
            <div class="score-fill {tone}" style="width: {safe_value * 100:.1f}%;"></div>
        </div>
    </div>
    """


ensure_state()

# Sidebar: model and collection controls
with st.sidebar:
    st.markdown("### Search Stack")
    use_detector = st.checkbox(
        "Enable object detection",
        value=True,
        help="Uses Grounding DINO to verify objects, counts, and rough positions.",
    )
    use_verifier = st.checkbox(
        "Enable caption verification",
        value=True,
        help="Uses Florence-2 to cross-check scene meaning with generated captions.",
    )

pipeline, pipeline_load_seconds = run_with_loader(
    "Preparing models and index...",
    load_pipeline,
    use_detector,
    use_verifier,
)

current_pipeline_flags = (use_detector, use_verifier)
if st.session_state.pipeline_flags is None:
    st.session_state.pipeline_flags = current_pipeline_flags
elif st.session_state.pipeline_flags != current_pipeline_flags:
    clear_search_state()
    st.session_state.pipeline_flags = current_pipeline_flags

with st.sidebar:
    st.caption(
        f"Pipeline ready in {pipeline_load_seconds:.1f}s. "
        f"Detector: {format_stage_state(use_detector)}. "
        f"Verifier: {format_stage_state(use_verifier)}."
    )

    st.markdown("### Image Collection")
    new_folder_path = st.text_input(
        "Folder path",
        value=st.session_state.folder_path,
        placeholder="Point to a folder with image files",
        help="The app scans this folder recursively for JPG, PNG, BMP, and WEBP files.",
    )
    st.session_state.folder_path = new_folder_path.strip() or "."
    folder_path = st.session_state.folder_path

    browse_clicked = st.button("Browse folders", use_container_width=True)
    if browse_clicked:
        selected = select_folder()
        if selected:
            st.session_state.folder_path = selected
            st.rerun()

    build_col, reload_col = st.columns(2)
    with build_col:
        build_button = st.button("Build index", use_container_width=True)
    with reload_col:
        reload_button = st.button("Reload index", use_container_width=True)

    if pipeline.search_engine.index is not None:
        st.success(f"Index loaded for {len(pipeline.search_engine.image_paths)} images")
    else:
        st.warning("No saved index is loaded yet")

    st.caption(f"Current folder: `{folder_path}`")
    st.caption(f"Saved index files: `{INDEX_PATH}` and `{METADATA_PATH}`")

if build_button:
    if os.path.exists(folder_path):
        def build_and_save():
            total_images = pipeline.build_index(folder_path)
            pipeline.save_index(INDEX_PATH, METADATA_PATH)
            return total_images

        try:
            total, build_seconds = run_with_loader(
                "Building the image index...",
                build_and_save,
            )
            clear_search_state()
            st.session_state.collection_notice = (
                f"Indexed {total} images from `{folder_path}` in {build_seconds:.1f}s."
            )
        except Exception as exc:
            st.session_state.collection_notice = None
            with st.sidebar:
                st.error(f"Index build failed: {exc}")
                st.exception(exc)
    else:
        with st.sidebar:
            st.error("Folder does not exist.")

if reload_button:
    try:
        _, reload_seconds = run_with_loader(
            "Reloading the saved index...",
            pipeline.load_index,
            INDEX_PATH,
            METADATA_PATH,
        )
        clear_search_state()
        st.session_state.collection_notice = (
            f"Loaded {len(pipeline.search_engine.image_paths)} indexed images in "
            f"{reload_seconds:.1f}s."
        )
    except Exception as exc:
        st.session_state.collection_notice = None
        with st.sidebar:
            st.error(f"Could not reload the saved index: {exc}")
            st.exception(exc)

image_count = len(pipeline.search_engine.image_paths) if pipeline.search_engine.index is not None else 0
active_stage_count = 1 + int(use_detector) + int(use_verifier)
index_state = "Ready" if image_count else "Waiting for index"

st.markdown(
    f"""
    <section class="hero-shell">
        <span class="eyebrow">Local multi-stage retrieval</span>
        <h1 class="hero-title">Find images by meaning, objects, and scene details.</h1>
        <p class="hero-copy">
            Search your local collection in natural language, then refine the ranking with
            semantic similarity, object detection, and caption-based scene verification.
        </p>
        <div class="hero-stat-grid">
            <div class="hero-stat">
                <div class="hero-stat-label">Collection status</div>
                <div class="hero-stat-value">{escape(index_state)}</div>
            </div>
            <div class="hero-stat">
                <div class="hero-stat-label">Indexed images</div>
                <div class="hero-stat-value">{image_count}</div>
            </div>
            <div class="hero-stat">
                <div class="hero-stat-label">Active stages</div>
                <div class="hero-stat-value">{active_stage_count}</div>
            </div>
            <div class="hero-stat">
                <div class="hero-stat-label">Current folder</div>
                <div class="hero-stat-value">{escape(os.path.basename(folder_path) or folder_path)}</div>
            </div>
        </div>
    </section>
    """,
    unsafe_allow_html=True,
)

if st.session_state.collection_notice:
    st.success(st.session_state.collection_notice)

guide_cols = st.columns(3)
guide_cards = [
    (
        "Stage 1: Semantic recall",
        "CLIP retrieves visually similar candidates from the whole library before any slower checks run.",
    ),
    (
        "Stage 2: Object checks",
        "Grounding DINO validates objects, counts, and loose position cues like left, right, or center.",
    ),
    (
        "Stage 3: Scene confidence",
        "Florence-2 compares generated captions with the parsed query so final rankings stay more readable.",
    ),
]

for column, (title, description) in zip(guide_cols, guide_cards):
    with column:
        st.markdown(
            f"""
            <div class="stage-card">
                <h4>{escape(title)}</h4>
                <p>{escape(description)}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown('<div class="section-title">Search</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-note">Start with a plain-language description. Use the advanced panel only when you need tighter control over ranking behavior.</div>',
    unsafe_allow_html=True,
)

st.caption("Quick prompts")
for row_start in range(0, len(SUGGESTED_QUERIES), 2):
    prompt_cols = st.columns(2)
    for offset, suggestion in enumerate(SUGGESTED_QUERIES[row_start:row_start + 2]):
        with prompt_cols[offset]:
            index = row_start + offset
            if st.button(suggestion, key=f"suggestion_{index}", use_container_width=True):
                st.session_state.query_input = suggestion
                st.rerun()

with st.form("search_form"):
    query = st.text_input(
        "Describe the image you want to find",
        key="query_input",
        placeholder="Example: three trees on the left side and a bird flying above them",
    )

    with st.expander("Advanced ranking controls", expanded=False):
        setting_col1, setting_col2 = st.columns(2)
        with setting_col1:
            top_k = st.slider("Results to show", min_value=1, max_value=12, value=6)
            candidate_k = st.slider("Stage 1 candidates", min_value=5, max_value=60, value=24)
        with setting_col2:
            clip_weight = st.slider("CLIP weight", min_value=0.0, max_value=1.0, value=0.4, step=0.05)
            st.caption("Detection and caption stages keep a fixed weight of 0.30 each when enabled.")

    search_submitted = st.form_submit_button(
        "Search library",
        type="primary",
        use_container_width=True,
        disabled=pipeline.search_engine.index is None,
    )

if pipeline.search_engine.index is None:
    st.info("Build or reload an index to enable searching.")

if search_submitted:
    detection_weight = 0.3
    caption_weight = 0.3

    if not query.strip():
        st.session_state.search_results = None
        st.session_state.parsed_query = None
        st.session_state.search_error = "Please enter a search prompt."
        st.session_state.search_traceback = None
    else:
        try:
            (results, parsed_query), search_seconds = run_with_loader(
                "Searching your library...",
                pipeline.search,
                query=query.strip(),
                top_k=top_k,
                candidate_k=candidate_k,
                clip_weight=clip_weight,
                detection_weight=detection_weight,
                caption_weight=caption_weight,
            )
            st.session_state.search_results = results
            st.session_state.parsed_query = parsed_query
            st.session_state.search_error = None
            st.session_state.search_traceback = None
            st.session_state.last_query = query.strip()
            st.session_state.last_search_seconds = search_seconds
        except Exception as exc:
            st.session_state.search_results = None
            st.session_state.parsed_query = None
            st.session_state.search_error = f"Search failed: {exc}"
            st.session_state.search_traceback = traceback.format_exc()

if st.session_state.search_error:
    st.error(st.session_state.search_error)
    if st.session_state.search_traceback:
        with st.expander("Technical details", expanded=False):
            st.code(st.session_state.search_traceback)

results = st.session_state.search_results
parsed_query = st.session_state.parsed_query

if results is not None:
    summary_col1, summary_col2, summary_col3 = st.columns(3)
    with summary_col1:
        st.metric("Matches", len(results))
    with summary_col2:
        st.metric("Last query", st.session_state.last_query or "-")
    with summary_col3:
        elapsed_text = (
            f"{st.session_state.last_search_seconds:.1f}s"
            if st.session_state.last_search_seconds is not None
            else "-"
        )
        st.metric("Search time", elapsed_text)

    chips = format_query_conditions(parsed_query)
    if chips:
        st.markdown('<div class="section-title">Parsed Conditions</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="chip-row">' + "".join(
                f'<span class="chip">{escape(chip)}</span>' for chip in chips
            ) + "</div>",
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-title">Results</div>', unsafe_allow_html=True)

    if results:
        result_columns = st.columns(2)
        for idx, result in enumerate(results):
            target_column = result_columns[idx % 2]
            with target_column:
                st.image(
                    result["image_path"],
                    caption=f"Rank #{idx + 1}  |  Final score {result['final_score']:.3f}",
                    use_container_width=True,
                )

                st.markdown(
                    f"""
                    <div class="result-header">
                        <div>
                            <div class="result-name">{escape(os.path.basename(result["image_path"]))}</div>
                            <div class="result-path">{escape(result["image_path"])}</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                metric_col1, metric_col2 = st.columns(2)
                with metric_col1:
                    st.metric("Final", f"{result['final_score']:.3f}")
                with metric_col2:
                    st.metric("CLIP", f"{result['clip_score']:.3f}")

                metric_col3, metric_col4 = st.columns(2)
                with metric_col3:
                    st.metric("Detection", f"{result['detection_score']:.3f}")
                with metric_col4:
                    st.metric("Caption", f"{result['caption_score']:.3f}")

                st.markdown(
                    """
                    <div class="score-stack">
                    """
                    + render_score_meter("Final score", result["final_score"], "final")
                    + render_score_meter("CLIP similarity", result["clip_score"], "clip")
                    + render_score_meter("Detection match", result["detection_score"], "detect")
                    + render_score_meter("Caption match", result["caption_score"], "caption")
                    + """
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if result.get("detection_details"):
                    with st.expander("Detection details", expanded=False):
                        for detail in result["detection_details"]:
                            st.write(detail)

                if result.get("caption"):
                    with st.expander("Generated caption", expanded=False):
                        st.write(result["caption"])

                if result.get("caption_details"):
                    with st.expander("Caption verification", expanded=False):
                        for detail in result["caption_details"]:
                            st.write(detail)

                st.divider()
    else:
        st.markdown(
            """
            <div class="empty-state">
                No matching images were found. Try a broader description, a lower CLIP weight,
                or rebuild the index after adding more images to the folder.
            </div>
            """,
            unsafe_allow_html=True,
        )
elif not st.session_state.search_error:
    st.markdown(
        """
        <div class="empty-state">
            Search results will appear here. Start with a short scene description, then refine
            the ranking controls only if the first pass is too broad or too strict.
        </div>
        """,
        unsafe_allow_html=True,
    )
