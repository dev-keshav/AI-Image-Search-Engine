"""
pipeline.py

Multi-stage image search pipeline:
  Stage 1: OpenCLIP + FAISS → semantic retrieval (top N candidates)
  Stage 2: Grounding DINO → object detection, counting, spatial verification
  Stage 3: Florence-2 → caption-based scene verification
  Final: Combined scoring and ranking
"""

from search_engine import ImageSearchEngine
from query_parser import parse_query, get_detection_labels
from object_detector import ObjectDetector
from scene_verifier import SceneVerifier


class SearchPipeline:
    def __init__(self, use_detector=True, use_verifier=True):
        """
        Initialize the multi-stage search pipeline.

        Args:
            use_detector: whether to load Grounding DINO (Stage 2)
            use_verifier: whether to load Florence-2 (Stage 3)
        """
        # Stage 1: Semantic retrieval
        print("=" * 50)
        print("Loading Stage 1: OpenCLIP + FAISS")
        print("=" * 50)
        self.search_engine = ImageSearchEngine()

        # Stage 2: Object detection
        self.detector = None
        if use_detector:
            print("=" * 50)
            print("Loading Stage 2: Grounding DINO")
            print("=" * 50)
            self.detector = ObjectDetector()

        # Stage 3: Scene verification
        self.verifier = None
        if use_verifier:
            print("=" * 50)
            print("Loading Stage 3: Florence-2")
            print("=" * 50)
            self.verifier = SceneVerifier()

        print("=" * 50)
        print("Pipeline ready!")
        print("=" * 50)

    def build_index(self, folder_path):
        """Build FAISS index for images in folder."""
        return self.search_engine.build_index(folder_path)

    def save_index(self, index_path="data/faiss_index.bin", metadata_path="data/metadata.pkl"):
        self.search_engine.save_index(index_path, metadata_path)

    def load_index(self, index_path="data/faiss_index.bin", metadata_path="data/metadata.pkl"):
        self.search_engine.load_index(index_path, metadata_path)

    def search(self, query, top_k=5, candidate_k=20,
               clip_weight=0.4, detection_weight=0.3, caption_weight=0.3):
        """
        Full multi-stage search.

        Args:
            query: natural language prompt
            top_k: number of final results to return
            candidate_k: number of candidates from Stage 1
            clip_weight: weight for semantic similarity score
            detection_weight: weight for object detection score
            caption_weight: weight for caption verification score

        Returns:
            list of results sorted by combined score
        """
        # Parse query into structured conditions
        parsed = parse_query(query)
        parsed_objects = parsed["objects"]

        # Stage 1: Semantic retrieval
        candidates = self.search_engine.search(
            query=query,
            top_k=candidate_k,
            fetch_k=candidate_k,
            min_score=0.10,
            relative_threshold=0.70
        )

        if not candidates:
            return [], parsed

        results = []

        for candidate in candidates:
            image_path = candidate["image_path"]
            clip_score = candidate["score"]

            detection_score = 0.0
            detection_details = []
            caption_score = 0.0
            caption_details = []
            caption_text = ""

            # Stage 2: Object detection verification
            if self.detector and parsed_objects:
                try:
                    det_result = self.detector.verify_conditions(
                        image_path, parsed_objects
                    )
                    detection_score = det_result["match_score"]
                    detection_details = det_result["details"]
                except Exception as e:
                    detection_details = [f"Detection error: {e}"]

            # Stage 3: Caption verification
            if self.verifier and parsed_objects:
                try:
                    ver_result = self.verifier.verify_image(
                        image_path, parsed_objects
                    )
                    caption_score = ver_result["match_score"]
                    caption_details = ver_result["details"]
                    caption_text = ver_result.get("caption", "")
                except Exception as e:
                    caption_details = [f"Verification error: {e}"]

            # Compute weights based on which stages are active
            active_weight_sum = clip_weight
            if self.detector and parsed_objects:
                active_weight_sum += detection_weight
            if self.verifier and parsed_objects:
                active_weight_sum += caption_weight

            # Normalize weights
            w_clip = clip_weight / active_weight_sum
            w_det = detection_weight / active_weight_sum if (self.detector and parsed_objects) else 0
            w_cap = caption_weight / active_weight_sum if (self.verifier and parsed_objects) else 0

            # Combined score
            final_score = (
                w_clip * clip_score +
                w_det * detection_score +
                w_cap * caption_score
            )

            results.append({
                "image_path": image_path,
                "final_score": final_score,
                "clip_score": clip_score,
                "detection_score": detection_score,
                "caption_score": caption_score,
                "detection_details": detection_details,
                "caption_details": caption_details,
                "caption": caption_text,
            })

        # Sort by final score
        results.sort(key=lambda x: x["final_score"], reverse=True)

        return results[:top_k], parsed