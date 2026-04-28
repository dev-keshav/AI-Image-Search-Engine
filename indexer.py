"""
indexer.py

Build and save the FAISS image index.
Only uses OpenCLIP (Stage 1). Detection/verification not needed for indexing.
"""

from search_engine import ImageSearchEngine

if __name__ == "__main__":
    folder_path = input("Enter the folder path containing images: ").strip()

    engine = ImageSearchEngine()

    print(f"Indexing images from: {folder_path}")
    total_images = engine.build_index(folder_path)
    print(f"Indexed {total_images} images successfully.")

    engine.save_index()
    print("Index and metadata saved to data/ folder.")