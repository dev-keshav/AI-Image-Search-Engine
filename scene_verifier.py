"""
scene_verifier.py

Uses Florence-2 for:
  - Image captioning (detailed scene descriptions)
  - Visual Question Answering (verify specific conditions)

Acts as the final verification stage in the pipeline.
"""

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM


class SceneVerifier:
    def __init__(self, model_id="microsoft/Florence-2-base", torch_dtype=None):
        """
        Load Florence-2 model and processor.
        Use 'Florence-2-base' for speed or 'Florence-2-large' for accuracy.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if torch_dtype is None:
            self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        else:
            self.torch_dtype = torch_dtype

        print(f"Loading Florence-2 on {self.device}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=self.torch_dtype,
            trust_remote_code=True
        ).to(self.device)

        self.processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True
        )
        self.model.eval()
        print("Florence-2 loaded.")

    def _run_task(self, image_path, task_prompt, text_input=None):
        """
        Run a Florence-2 task on an image.

        Args:
            image_path: path to image
            task_prompt: Florence-2 task token (e.g. '<CAPTION>', '<DETAILED_CAPTION>', '<MORE_DETAILED_CAPTION>')
            text_input: optional additional text input

        Returns:
            parsed result from Florence-2
        """
        image = Image.open(image_path).convert("RGB")

        if text_input:
            prompt = task_prompt + text_input
        else:
            prompt = task_prompt

        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        ).to(self.device, self.torch_dtype)

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
            )

        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]

        parsed = self.processor.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=image.size
        )

        return parsed

    def caption(self, image_path, detail_level="detailed"):
        """
        Generate a caption for an image.

        Args:
            detail_level: "basic", "detailed", or "more_detailed"

        Returns:
            caption string
        """
        task_map = {
            "basic": "<CAPTION>",
            "detailed": "<DETAILED_CAPTION>",
            "more_detailed": "<MORE_DETAILED_CAPTION>",
        }

        task_prompt = task_map.get(detail_level, "<DETAILED_CAPTION>")
        result = self._run_task(image_path, task_prompt)

        # Extract caption text from result
        if isinstance(result, dict):
            # Result may have the task key
            for key in result:
                if isinstance(result[key], str):
                    return result[key]
            return str(result)
        return str(result)

    def verify_caption_match(self, caption, parsed_objects):
        """
        Check how well the caption matches the parsed query conditions.

        Simple keyword matching approach.

        Returns:
            {
                "total_checks": int,
                "passed_checks": int,
                "match_score": float,
                "details": [...]
            }
        """
        caption_lower = caption.lower()
        total_checks = 0
        passed_checks = 0
        details = []

        for obj in parsed_objects:
            name = obj.get("name", "")
            if not name:
                continue

            # Check if object name appears in caption
            total_checks += 1
            name_lower = name.lower()

            # Check for the name or parts of it
            name_parts = name_lower.split()
            found = False
            for part in name_parts:
                if len(part) > 2 and part in caption_lower:
                    found = True
                    break

            if found:
                passed_checks += 1
                details.append(f"✅ Caption mentions '{name}'")
            else:
                details.append(f"❌ Caption does not mention '{name}'")

            # Check action in caption
            action = obj.get("action")
            if action:
                total_checks += 1
                if action.lower() in caption_lower:
                    passed_checks += 1
                    details.append(f"✅ Caption mentions '{action}'")
                else:
                    details.append(f"❌ Caption does not mention '{action}'")

        match_score = passed_checks / total_checks if total_checks > 0 else 0.0

        return {
            "total_checks": total_checks,
            "passed_checks": passed_checks,
            "match_score": match_score,
            "details": details,
            "caption": caption
        }

    def verify_image(self, image_path, parsed_objects):
        """
        Full verification of an image against parsed conditions.
        Generates a detailed caption and checks against conditions.

        Returns verification result dict.
        """
        caption = self.caption(image_path, detail_level="more_detailed")
        result = self.verify_caption_match(caption, parsed_objects)
        return result