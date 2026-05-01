"""
object_detector.py

Uses Grounding DINO for open-vocabulary object detection.
Detects objects by text prompt, returns bounding boxes.
Used for:
  - object presence verification
  - object counting
  - spatial position checking
"""

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection


class ObjectDetector:
    def __init__(self, model_id="IDEA-Research/grounding-dino-tiny"):
        """
        Load Grounding DINO model and processor.
        Uses the tiny variant for speed. Use 'grounding-dino-base' for accuracy.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading Grounding DINO on {self.device}...")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_id
        ).to(self.device)
        self.model.eval()
        print("Grounding DINO loaded.")

    def detect(self, image_path, labels, box_threshold=0.25, text_threshold=0.25):
        """
        Detect objects in an image using text labels.

        Args:
            image_path: path to the image file
            labels: list of object names, e.g. ["bird", "tree"]
            box_threshold: confidence threshold for boxes
            text_threshold: confidence threshold for text matching

        Returns:
            list of detections, each with:
              - label: detected object name
              - score: confidence score
              - box: [x1, y1, x2, y2] in pixels
        """
        image = Image.open(image_path).convert("RGB")
        image_width, image_height = image.size

        # Grounding DINO expects labels separated by periods
        # IMPORTANT: queries must be lowercase and end with a dot
        text = ". ".join([l.lower() for l in labels]) + "."

        inputs = self.processor(
            images=image,
            text=text,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[(image_height, image_width)]
        )[0]

        detections = []
        for score, label, box in zip(
            results["scores"],
            results["text_labels"],
            results["boxes"]
        ):
            box = box.cpu().tolist()
            detections.append({
                "label": label.strip(),
                "score": float(score),
                "box": box  # [x1, y1, x2, y2]
            })

        return detections, image_width, image_height

    def count_objects(self, detections, target_label):
        """
        Count how many detections match a target label.
        """
        count = 0
        for det in detections:
            if target_label.lower() in det["label"].lower():
                count += 1
        return count

    def check_spatial(self, detections, target_label, expected_location, image_width, image_height):
        """
        Check if detected objects are in the expected spatial region.

        Regions:
          - "left":   x_center < image_width * 0.5
          - "right":  x_center > image_width * 0.5
          - "top":    y_center < image_height * 0.4
          - "bottom": y_center > image_height * 0.6
          - "center": middle area

        Returns:
            (matching_count, total_count)
        """
        matching = 0
        total = 0

        for det in detections:
            if target_label.lower() not in det["label"].lower():
                continue

            total += 1

            x1, y1, x2, y2 = det["box"]
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2

            in_region = False

            if expected_location == "left":
                in_region = x_center < image_width * 0.5
            elif expected_location == "right":
                in_region = x_center > image_width * 0.5
            elif expected_location == "top":
                in_region = y_center < image_height * 0.4
            elif expected_location == "bottom":
                in_region = y_center > image_height * 0.6
            elif expected_location == "center":
                in_region = (
                    image_width * 0.25 < x_center < image_width * 0.75 and
                    image_height * 0.25 < y_center < image_height * 0.75
                )

            if in_region:
                matching += 1

        return matching, total

    def verify_conditions(self, image_path, parsed_objects, box_threshold=0.25, text_threshold=0.25):
        """
        Full verification: detect objects, check counts and spatial positions.

        Args:
            image_path: path to image
            parsed_objects: list of parsed object conditions from query_parser

        Returns:
            {
                "total_conditions": int,
                "passed_conditions": int,
                "match_score": float (0.0 to 1.0),
                "details": [...]
            }
        """
        # Collect all unique object names to detect
        labels = list(set([obj["name"] for obj in parsed_objects if obj["name"]]))

        if not labels:
            return {
                "total_conditions": 0,
                "passed_conditions": 0,
                "match_score": 0.0,
                "details": []
            }

        # Run detection once for all labels
        detections, img_w, img_h = self.detect(
            image_path, labels, box_threshold, text_threshold
        )

        total_conditions = 0
        passed_conditions = 0
        details = []

        for obj in parsed_objects:
            name = obj["name"]
            if not name:
                continue

            expected_count = obj.get("count")
            expected_location = obj.get("location")

            # Check presence
            total_conditions += 1
            actual_count = self.count_objects(detections, name)
            present = actual_count > 0

            if present:
                passed_conditions += 1
                details.append(f"✅ '{name}' found ({actual_count} detected)")
            else:
                details.append(f"❌ '{name}' not found")
                continue  # Skip further checks if not found

            # Check count
            if expected_count and expected_count > 1:
                total_conditions += 1
                if actual_count == expected_count:
                    passed_conditions += 1
                    details.append(f"✅ Count of '{name}' matches: {actual_count}")
                else:
                    details.append(
                        f"❌ Expected {expected_count} '{name}', found {actual_count}"
                    )

            # Check spatial position
            if expected_location:
                total_conditions += 1
                matching, total = self.check_spatial(
                    detections, name, expected_location, img_w, img_h
                )
                if matching > 0:
                    passed_conditions += 1
                    details.append(
                        f"✅ '{name}' found on {expected_location} ({matching}/{total})"
                    )
                else:
                    details.append(
                        f"❌ '{name}' not found on {expected_location}"
                    )

        match_score = passed_conditions / total_conditions if total_conditions > 0 else 0.0

        return {
            "total_conditions": total_conditions,
            "passed_conditions": passed_conditions,
            "match_score": match_score,
            "details": details,
            "detections": detections
        }