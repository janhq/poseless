import io
import json
import os
import time

import mujoco
import numpy as np
from datasets import Dataset, Features
from datasets import Image as DsImage
from datasets import Value
from huggingface_hub import HfApi
from PIL import Image
from tqdm.auto import tqdm

# Set up the model and data
model = mujoco.MjModel.from_xml_path("shadow_hand/left_hand.xml")
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, height=1280, width=1280)
joint_names = [
    "lh_WRJ2",
    "lh_WRJ1",
    "lh_FFJ4",
    "lh_FFJ3",
    "lh_FFJ2",
    "lh_FFJ1",
    "lh_MFJ4",
    "lh_MFJ3",
    "lh_MFJ2",
    "lh_MFJ1",
    "lh_RFJ4",
    "lh_RFJ3",
    "lh_RFJ2",
    "lh_RFJ1",
    "lh_LFJ5",
    "lh_LFJ4",
    "lh_LFJ3",
    "lh_LFJ2",
    "lh_LFJ1",
    "lh_THJ5",
    "lh_THJ4",
    "lh_THJ3",
    "lh_THJ2",
    "lh_THJ1",
]
SYSTEM_PROMPT = """You are a specialized Vision Language Model designed to accurately estimate joint angles from hand pose images. Your task is to analyze images of a human or robotic hand and output precise angle measurements for each joint. Output joint angles in radians.
Output Format:
<lh_WRJ2>angle</lh_WRJ2><lh_WRJ1>angle</lh_WRJ1><lh_FFJ4>angle</lh_FFJ4><lh_FFJ3>angle</lh_FFJ3><lh_FFJ2>angle</lh_FFJ2><lh_FFJ1>angle</lh_FFJ1><lh_MFJ4>angle</lh_MFJ4><lh_MFJ3>angle</lh_MFJ3><lh_MFJ2>angle</lh_MFJ2><lh_MFJ1>angle</lh_MFJ1><lh_RFJ4>angle</lh_RFJ4><lh_RFJ3>angle</lh_RFJ3><lh_RFJ2>angle</lh_RFJ2><lh_RFJ1>angle</lh_RFJ1><lh_LFJ5>angle</lh_LFJ5><lh_LFJ4>angle</lh_LFJ4><lh_LFJ3>angle</lh_LFJ3><lh_LFJ2>angle</lh_LFJ2><lh_LFJ1>angle</lh_LFJ1><lh_THJ5>angle</lh_THJ5><lh_THJ4>angle</lh_THJ4><lh_THJ3>angle</lh_THJ3><lh_THJ2>angle</lh_THJ2><lh_THJ1>angle</lh_THJ1>
"""
joint_name_to_index = {name: i for i, name in enumerate(joint_names)}


def get_n_pose_and_upload(
    n, dataset_name="hand-poses-dataset", push_to_hub=True, num_test_sample=1000
):
    assert (
        num_test_sample < n
    ), "The number of test samples must be lower than the total synthetic subset"
    """Generate n random hand poses and upload to Hugging Face."""

    global joint_names, joint_name_to_index, SYSTEM_PROMPT

    os.makedirs("data", exist_ok=True)

    # Prepare data structures for the dataset
    images_data = []
    joint_positions = []
    filenames = []

    pose_num = 0
    pbar = tqdm(total=n, desc="Generating poses", unit="pose")
    while len(filenames) < n:
        mujoco.mj_resetData(model, data)

        # Generate random joint positions
        target_positions = []
        for i in range(2, model.njnt):
            joint = model.joint(i)
            target_pos = np.random.uniform(joint.range[0], joint.range[1])
            target_positions.append((joint.qposadr[0], target_pos))

        # Apply control to reach target positions
        for step in range(2000):
            for qpos_addr, target_pos in target_positions:
                current_pos = data.qpos[qpos_addr]
                kp = 10.0  # Proportional gain
                error = target_pos - current_pos
                data.qfrc_applied[qpos_addr] = kp * error

            # Step physics
            mujoco.mj_step(model, data)

            # Check if close enough to targets
            if step % 100 == 0:
                total_error = 0
                for qpos_addr, target_pos in target_positions:
                    total_error += abs(target_pos - data.qpos[qpos_addr])
                if total_error < 0.1 or (step > 500 and data.ncon < 10):
                    break

        # Forward kinematics and render
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera="closeup")
        pixels = renderer.render()

        buf = io.BytesIO()
        image = Image.fromarray(pixels)
        image.save(buf, format="PNG")
        buf.seek(0)

        image_filename = f"pose_{pose_num}.png"
        image_path = os.path.join("data", image_filename)
        image.save(image_path)

        joint_positions.append(data.qpos.copy().tolist())
        filenames.append(image_filename)

        buf.close()
        image.close()
        pose_num += 1
        pbar.update(1)  # Update the main progress bar

    pbar.close()  # Close the main progress bar
    print("Processing image data and creating conversations...")

    # Create conversations with tqdm progress
    conversations = []
    for i in tqdm(range(len(filenames)), desc="Creating conversations", unit="conv"):
        # Format each joint angle with the special token format
        joint_description = ""
        for name in joint_names:
            if name in joint_name_to_index:
                joint_idx = joint_name_to_index[name]
                if joint_idx < len(joint_positions[i]):
                    angle_value = round(joint_positions[i][joint_idx], 2)
                    joint_description += f"<{name}>{angle_value}</{name}>"
        conversation = [
            {"role": "system", "content": f"{SYSTEM_PROMPT}"},
            {
                "role": "user",
                "content": [
                    {"type": "image", "content": f"data/{filenames[i]}"},
                    {"type": "text", "content": "<Pose>"},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "content": f"{joint_description}"},
                ],
            },
        ]
        conversations.append(conversation)

    conversations_json = [json.dumps(conv) for conv in conversations]
    output_path = "data/conversations_dataset.jsonl"
    with open(output_path, "w") as f:
        for conv_json in conversations_json:
            f.write(conv_json + "\n")

    for image_path in filenames:
        images_data.append(Image.open(f"data/{image_path}"))

    dataset_dict = {
        "image": images_data,
        "conversations": conversations_json,
    }

    # Create the Hugging Face dataset
    features = Features(
        {
            "image": DsImage(),
            "conversations": Value("string"),
        }
    )

    print("Building dataset object...")
    dataset = Dataset.from_dict(dataset_dict, features=features)

    # Save metadata as JSON
    print("Creating metadata...")
    model_joint_names = [model.joint(i).name for i in range(model.njnt)]
    metadata = {
        "joint_names": model_joint_names,
        "dataset_description": "Random hand poses generated with MuJoCo",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_poses": n,
    }

    with open(os.path.join("data", "_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    if push_to_hub:
        try:
            print(f"Pushing dataset to Hugging Face Hub as {dataset_name}")
            dataset = dataset.train_test_split(test_size=num_test_sample)
            dataset.push_to_hub(
                dataset_name,
            )

            # push the metadata
            api = HfApi()
            print("Uploading metadata...")
            api.upload_file(
                path_or_fileobj=os.path.join("data", "_metadata.json"),
                path_in_repo="_metadata.json",
                repo_id=dataset_name,
                repo_type="dataset",
            )
            print("Upload successful!")
        except Exception as e:
            print(f"Error uploading to Hugging Face Hub: {e}")
            print("Saving dataset locally instead")
            dataset.save_to_disk("data/hf_dataset")
    else:
        print("Saving dataset locally")
        with tqdm(total=100, desc="Saving locally", unit="%") as pbar:
            dataset.save_to_disk("data/hf_dataset")
            pbar.update(100)

    # Clean up resources
    print("Cleaning up resources...")
    for img in images_data:
        try:
            img.close()
        except:
            pass

    return dataset


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate n random hand poses and upload to Hugging Face."
    )
    parser.add_argument("n", type=int, help="Number of poses to generate")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="jan-hq/robotic-hand-poses",
        help="Name of the Hugging Face dataset to create",
    )
    parser.add_argument(
        "--no_push",
        action="store_true",
        help="Don't push to Hugging Face, just save locally",
    )
    parser.add_argument(
        "--num_test_samples",
        type=int,
        default="1000",
        help="Number of test samples to split",
    )

    args = parser.parse_args()

    print(f"Starting generation of {args.n} hand poses...")
    start_time = time.time()
    get_n_pose_and_upload(
        args.n, args.dataset_name, not args.no_push, args.num_test_samples
    )
    end_time = time.time()

    elapsed_time = end_time - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    print(f"Time elapsed: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
