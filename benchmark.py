import re
import torch
from qwen_vl_utils import process_vision_info
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)

SYSTEM_PROMPT = """You are a specialized Vision Language Model designed to accurately estimate joint angles from hand pose images. Your task is to analyze images of a human or robotic hand and output precise angle measurements for each joint. Output joint angles in radians.
Output Format:
<lh_WRJ2>angle</lh_WRJ2><lh_WRJ1>angle</lh_WRJ1><lh_FFJ4>angle</lh_FFJ4><lh_FFJ3>angle</lh_FFJ3><lh_FFJ2>angle</lh_FFJ2><lh_FFJ1>angle</lh_FFJ1><lh_MFJ4>angle</lh_MFJ4><lh_MFJ3>angle</lh_MFJ3><lh_MFJ2>angle</lh_MFJ2><lh_MFJ1>angle</lh_MFJ1><lh_RFJ4>angle</lh_RFJ4><lh_RFJ3>angle</lh_RFJ3><lh_RFJ2>angle</lh_RFJ2><lh_RFJ1>angle</lh_RFJ1><lh_LFJ5>angle</lh_LFJ5><lh_LFJ4>angle</lh_LFJ4><lh_LFJ3>angle</lh_LFJ3><lh_LFJ2>angle</lh_LFJ2><lh_LFJ1>angle</lh_LFJ1><lh_THJ5>angle</lh_THJ5><lh_THJ4>angle</lh_THJ4><lh_THJ3>angle</lh_THJ3><lh_THJ2>angle</lh_THJ2><lh_THJ1>angle</lh_THJ1>
"""


class HandPoseProcessor:
    def __init__(self, model_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.min_pixels = 256 * 28 * 28
        self.max_pixels = 1280 * 28 * 28

        self.model = (
            Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            )
            .eval()
            .to(self.device)
        )

        self.processor = AutoProcessor.from_pretrained(
            model_path,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
            trust_remote_code=True,
        )

    def parse_angles(self, xml_string):
        pattern = r"<([^>]+)>([^<]+)</\1>"
        matches = re.findall(pattern, xml_string)
        angles = []
        angle_dict = {}

        for tag, value in matches:
            try:
                float_value = float(value)
                angles.append(float_value)
                angle_dict[tag] = float_value
            except ValueError:
                print(
                    f"Error: Could not convert value '{value}' for tag '{tag}' to float"
                )
                angles.append(None)
                angle_dict[tag] = None

        return angles, angle_dict

    def process_image(self, image_path):
        vlm_output = self._infer_angles(image_path)
        return vlm_output

    def _infer_angles(self, image_path):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                        "min_pixels": 1003520,
                        "max_pixels": 1003520,
                    },
                    {"type": "text", "text": "<Pose>"},
                ],
            },
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        inputs = inputs.to(self.device)
        generated_ids = self.model.generate(**inputs, max_new_tokens=1024)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return output_text[0]


def main():
    processor = HandPoseProcessor("jan-hq/Poseless-3B-cp-1500")
    angles = processor.process_image("data/pose_0.png")
    print("Joint angles:", angles)


if __name__ == "__main__":
    main()
