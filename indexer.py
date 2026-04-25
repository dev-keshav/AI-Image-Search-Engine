from search_engine import ImageSearchEngine

if __name__ == "__main__":
    folder_path = input("Enter the folder path containing images: ").strip()

    engine = ImageSearchEngine()

    total_images = engine.build_index(folder_path)
    print(f"Indexed {total_images} images successfully.")

    engine.save_index()
    print("Index and metadata saved successfully in the data/ folder.")