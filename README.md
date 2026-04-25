# Local AI Image Search Engine

A local semantic image search engine built with **OpenCLIP**, **FAISS**, and **Streamlit**.

## Features
- Search local images using natural language prompts
- Fast retrieval using vector similarity
- Build and save searchable image indexes
- Streamlit-based user interface

## Tech Stack
- Python
- OpenCLIP
- FAISS
- Streamlit
- PyTorch
- Pillow
- NumPy

## How it Works
1. Images are loaded from a local folder.
2. OpenCLIP generates semantic embeddings for each image.
3. Embeddings are stored in a FAISS index.
4. The user's text query is converted into an embedding.
5. FAISS returns the most similar images.

## Installation

```bash
git clone https://github.com/your-username/local-image-search.git
cd local-image-search
pip install -r requirements.txt