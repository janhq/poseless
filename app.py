import os
import re
from datetime import datetime

import gradio as gr
import mujoco
import numpy as np
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)

SYSTEM_PROMPT = """You are a specialized Vision Language Model designed to accurately estimate joint angles from hand pose images. Your task is to analyze images of a human or robotic hand and output precise angle measurements for each joint. Output joint angles in radians.
Output Format:
<lh_WRJ2>angle</lh_WRJ2><lh_WRJ1>angle</lh_WRJ1><lh_FFJ4>angle</lh_FFJ4><lh_FFJ3>angle</lh_FFJ3><lh_FFJ2>angle</lh_FFJ2><lh_FFJ1>angle</lh_FFJ1><lh_MFJ4>angle</lh_MFJ4><lh_MFJ3>angle</lh_MFJ3><lh_MFJ2>angle</lh_MFJ2><lh_MFJ1>angle</lh_MFJ1><lh_RFJ4>angle</lh_RFJ4><lh_RFJ3>angle</lh_RFJ3><lh_RFJ2>angle</lh_RFJ2><lh_RFJ1>angle</lh_RFJ1><lh_LFJ5>angle</lh_LFJ5><lh_LFJ4>angle</lh_LFJ4><lh_LFJ3>angle</lh_LFJ3><lh_LFJ2>angle</lh_LFJ2><lh_LFJ1>angle</lh_LFJ1><lh_THJ5>angle</lh_THJ5><lh_THJ4>angle</lh_THJ4><lh_THJ3>angle</lh_THJ3><lh_THJ2>angle</lh_THJ2><lh_THJ1>angle</lh_THJ1>
"""
device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "jan-hq/Poseless-3B-cp-1500"
min_pixels = 256 * 28 * 28
max_pixels = 1280 * 28 * 28
model = (
    Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    .eval()
    .to(device)
)
processor = AutoProcessor.from_pretrained(
    model_path, min_pixels=min_pixels, max_pixels=max_pixels, trust_remote_code=True
)


def parse_angles(xml_string):
    # Regular expression to match tags and their content
    pattern = r"<([^>]+)>([^<]+)</\1>"

    # Find all matches in the input string
    matches = re.findall(pattern, xml_string)

    # Extract the angle values and convert to float
    angles = []
    angle_dict = {}

    for tag, value in matches:
        try:
            float_value = float(value)
            angles.append(float_value)
            angle_dict[tag] = float_value
        except ValueError:
            print(f"Error: Could not convert value '{value}' for tag '{tag}' to float")
            angles.append(None)
            angle_dict[tag] = None

    return angles, angle_dict


# Copy from Qwen2.5 VL demo: https://huggingface.co/spaces/mrdbourke/Qwen2.5-VL-Instruct-Demo/blob/main/app.py.
def array_to_image_path(image_array):
    if image_array is None:
        raise ValueError("No image provided. Please upload an image before submitting.")
    # Convert numpy array to PIL Image
    img = Image.fromarray(np.uint8(image_array))

    # Generate a unique filename using timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"image_{timestamp}.png"

    # Save the image
    img.save(filename)

    # Get the full path of the saved image
    full_path = os.path.abspath(filename)

    return full_path


def poseless_infer(image, text_input="<Pose>", device=device):
    global SYSTEM_PROMPT, model, processor
    image_path = array_to_image_path(image)
    image = Image.fromarray(image).convert("RGB")
    messages = [
        {"role": "system", "content": f"{SYSTEM_PROMPT}"},
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                    "min_pixels": 1003520,
                    "max_pixels": 1003520,
                },
                {"type": "text", "text": text_input},
            ],
        },
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    print(text)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)
    generated_ids = model.generate(**inputs, max_new_tokens=1024)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    print(output_text[0])
    return output_text[0]


def render_hand_image(vlm_text):
    # Init robot hand using mujoco
    model_render = mujoco.MjModel.from_xml_path("shadow_hand/left_hand.xml")
    data = mujoco.MjData(model_render)
    renderer = mujoco.Renderer(model_render, height=1280, width=1280)
    joint_name_to_index = {
        model_render.joint(i).name: model_render.joint(i).qposadr[0]
        for i in range(model_render.njnt)
    }
    _, target_positions_dict = parse_angles(vlm_text)
    target_positions = [
        (joint_name_to_index[k], np.float32(v))
        for k, v in target_positions_dict.items()
    ]
    print(target_positions)
    # mujoco.mj_resetData(model_render, data)
    for step in range(2000):
        for qpos_addr, target_pos in target_positions:
            current_pos = data.qpos[qpos_addr]
            kp = 10.0  # Proportional gain
            error = target_pos - current_pos
            data.qfrc_applied[qpos_addr] = kp * error

        # Step physics
        mujoco.mj_step(model_render, data)
        if step % 100 == 0:
            total_error = 0
            for qpos_addr, target_pos in target_positions:
                total_error += abs(target_pos - data.qpos[qpos_addr])
            if total_error < 0.1 or (step > 500 and data.ncon < 10):
                break

    # Forward kinematics and render
    mujoco.mj_forward(model_render, data)
    renderer.update_scene(data, camera="closeup")
    pixels = renderer.render()
    image = Image.fromarray(pixels)
    return image


def process_hand_image(input_image):
    if input_image is None:
        return None, "No image provided"

    # Step 1: Process the image with VLM
    try:
        vlm_output = poseless_infer(input_image)  # Get first result from the list

        # Step 2: Render the hand using the angles from VLM output
        rendered_image = render_hand_image(vlm_output)

        # Return both the rendered image and the VLM output
        return rendered_image, vlm_output
    except Exception as e:
        return None, f"Error processing image: {str(e)}"
    return rendered_image, vlm_output


# Custom CSS for robotic theme
css = """
:root {
    --main-bg-color: #1a1a2e;
    --panel-bg-color: #16213e;
    --accent-color: #0f3460;
    --highlight-color: #00b4d8;
    --text-color: #e7e7e7;
}

body {
    background-color: var(--main-bg-color);
    color: var(--text-color);
}

.container {
    max-width: 1200px;
    margin: 0 auto;
}

.gr-box {
    border-radius: 15px;
    border: 2px solid var(--highlight-color);
    background-color: var(--panel-bg-color);
    box-shadow: 0 0 15px rgba(0, 180, 216, 0.3);
}

.gr-button {
    background-color: var(--accent-color);
    border: 2px solid var(--highlight-color);
    border-radius: 8px;
    color: white;
    font-weight: bold;
    transition: all 0.3s;
}

.gr-button:hover {
    background-color: var(--highlight-color);
    transform: scale(1.05);
    box-shadow: 0 0 10px rgba(0, 180, 216, 0.5);
}

.title {
    text-align: center;
    color: var(--highlight-color);
    text-shadow: 0 0 10px rgba(0, 180, 216, 0.5);
    font-family: 'Orbitron', sans-serif;
    margin-bottom: 20px;
}

.status-indicator {
    height: 10px;
    width: 10px;
    background-color: #4CAF50;
    border-radius: 50%;
    display: inline-block;
    margin-right: 10px;
}

.panel-title {
    border-bottom: 1px solid var(--highlight-color);
    padding-bottom: 8px;
    margin-bottom: 16px;
    font-family: 'Orbitron', sans-serif;
    color: var(--highlight-color);
}
"""
with gr.Blocks(css=css, title="POSELESS") as demo:
    # Header with robotic style
    gr.HTML(
        """
        <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&display=swap" rel="stylesheet">
        <div class="title">
            <h1>POSELESS v1.0</h1>
            <p><span class="status-indicator"></span> SYSTEM ONLINE</p>
        </div>
    """
    )

    # Main interface
    with gr.Row():
        # Input panel
        with gr.Column():
            gr.HTML('<div class="panel-title">INPUT TERMINAL</div>')
            input_image = gr.Image(label="Input Hand Image", height=400)

            with gr.Row():
                clear_btn = gr.Button("RESET", variant="secondary")
                process_btn = gr.Button("ANALYZE", variant="primary")

            gr.HTML(
                """
                <div style="margin-top: 20px; padding: 15px; border: 1px solid #00b4d8; border-radius: 8px; background-color: rgba(0, 180, 216, 0.1);">
                    <h4 style="margin-top: 0;">SYSTEM INSTRUCTIONS:</h4>
                    <ol>
                        <li>Upload a clear image of a hand</li>
                        <li>Press ANALYZE to process the image</li>
                        <li>View the rendered 3D model and angle data</li>
                    </ol>
                </div>
            """
            )

        # Output panel
        with gr.Column():
            gr.HTML('<div class="panel-title">OUTPUT TERMINAL</div>')
            output_image = gr.Image(label="3D MODEL RENDERING", height=400)
            with gr.Accordion("PROCESSING LOGS", open=False):
                output_text = gr.Textbox(
                    label="", lines=5, placeholder="Processing data will appear here..."
                )

            gr.HTML(
                """
                <div style="margin-top: 20px; font-family: monospace; padding: 10px; background-color: rgba(0, 180, 216, 0.1); border-radius: 8px;">
                    <div>SYSTEM STATUS: <span style="color: #4CAF50;">■ OPERATIONAL</span></div>
                    <div>MODEL: <span>HAND-POSE-VLM-v2.3</span></div>
                    <div>RENDERER: <span>MUJOCO-ENGINE-v1.5</span></div>
                </div>
            """
            )

    # Example section with robotic styling
    gr.HTML('<div class="panel-title" style="margin-top: 30px;">SAMPLE DATASETS</div>')
    gr.Examples(
        examples=["examples/pose_000000.png"],
        inputs=input_image,
        outputs=[output_image, output_text],
        fn=process_hand_image,
        examples_per_page=4,
    )

    # Event handlers
    process_btn.click(
        fn=process_hand_image, inputs=[input_image], outputs=[output_image, output_text]
    )

    clear_btn.click(
        fn=lambda: (None, ""), inputs=[], outputs=[output_image, output_text]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=True)
