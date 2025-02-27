import mujoco
from PIL import Image
import numpy as np
import json
import os
import time
from tqdm import tqdm
from datasets import Dataset, Features, Value, Image as DsImage
import threading
import concurrent.futures
import io  # Import the io module

MODEL_XML_PATH = "shadow_hand/left_hand.xml" 
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
  "lh_THJ1"
]
SYSTEM_PROMPT = """You are a specialized Vision Language Model designed to accurately estimate joint angles from hand pose images. Your task is to analyze images of a human or robotic hand and output precise angle measurements for each joint. Output joint angles in radians.
Output Format:
<lh_WRJ2>angle</lh_WRJ2><lh_WRJ1>angle</lh_WRJ1><lh_FFJ4>angle</lh_FFJ4><lh_FFJ3>angle</lh_FFJ3><lh_FFJ2>angle</lh_FFJ2><lh_FFJ1>angle</lh_FFJ1><lh_MFJ4>angle</lh_MFJ4><lh_MFJ3>angle</lh_MFJ3><lh_MFJ2>angle</lh_MFJ2><lh_MFJ1>angle</lh_MFJ1><lh_RFJ4>angle</lh_RFJ4><lh_RFJ3>angle</lh_RFJ3><lh_RFJ2>angle</lh_RFJ2><lh_RFJ1>angle</lh_RFJ1><lh_LFJ5>angle</lh_LFJ5><lh_LFJ4>angle</lh_LFJ4><lh_LFJ3>angle</lh_LFJ3><lh_LFJ2>angle</lh_LFJ2><lh_LFJ1>angle</lh_LFJ1><lh_THJ5>angle</lh_THJ5><lh_THJ4>angle</lh_THJ4><lh_THJ3>angle</lh_THJ3><lh_THJ2>angle</lh_THJ2><lh_THJ1>angle</lh_THJ1>
"""
joint_name_to_index = {name: i for i, name in enumerate(joint_names)}

def process_pose(model_xml_path, pose_num, seed=None):
    """Generates a single hand pose and returns the filename and joint positions.
    Each thread creates its own model, data, and renderer to avoid race conditions.
    """
    # Set random seed for reproducibility if provided
    if seed is not None:
        np.random.seed(seed + pose_num)
    
    model = mujoco.MjModel.from_xml_path(model_xml_path)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=1280, width=1280)
    
    mujoco.mj_resetData(model, data)

    # Generate random joint positions
    target_positions = []
    for i in range(2, model.njnt):
        joint = model.joint(i)
        target_pos = np.random.uniform(joint.range[0], joint.range[1])
        target_positions.append((joint.dofadr[0], target_pos))

    # Apply control to reach target positions
    for step in range(1000):
        kp = 10.0  # Proportional gain
        for dof_addr, target_pos in target_positions:
            jnt_id = model.dof_jntid[dof_addr]  # Correct joint ID.
            qpos_addr = model.jnt_qposadr[jnt_id]
            error = target_pos - data.qpos[qpos_addr]
            data.qfrc_applied[dof_addr] = kp * error  # Directly index qfrc_applied

        mujoco.mj_step(model, data)

        if step % 100 == 0:
            total_error = 0
            for dof_addr, target_pos in target_positions:
                jnt_id = model.dof_jntid[dof_addr]  # Correct joint ID
                qpos_addr = model.jnt_qposadr[jnt_id]
                total_error += abs(target_pos - data.qpos[qpos_addr])

            if total_error < 0.1 or (step > 500 and data.ncon > 10):
                break

    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera="closeup")

    pixels = renderer.render()

    # Save to BytesIO first to avoid race conditions writing to disk.
    buf = io.BytesIO()
    image = Image.fromarray(pixels)
    image.save(buf, format="PNG")
    buf.seek(0)

    image_filename = f"pose_{pose_num}.png"
    image_path = os.path.join("data", image_filename)
    with open(image_path, 'wb') as f:  
      f.write(buf.getbuffer())  

    # Clean up resources immediately
    renderer.close()
    
    joint_positions = data.qpos.copy().tolist()
    
    return image_filename, joint_positions

def get_n_pose_and_upload(n, dataset_name="hand-poses-dataset", push_to_hub=True, num_test_sample=1000, num_process=None):
    """Generate n random hand poses and upload to Hugging Face."""
    global joint_names, joint_name_to_index, SYSTEM_PROMPT
    assert num_test_sample < n, "The number of test samples must be lower than the total synthetic subset"

    # Load the model in the main thread just to get joint names
    model = mujoco.MjModel.from_xml_path(MODEL_XML_PATH)

    os.makedirs("data", exist_ok=True)

    # Prepare data structures for the dataset
    images_data = []
    joint_positions = []
    filenames = []

    pbar = tqdm(total=n, desc="Generating poses", unit="pose")

    # Use a base seed for reproducibility
    base_seed = int(time.time())
    
    num_threads = num_process if num_process is not None else os.cpu_count() 
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [
            executor.submit(process_pose, MODEL_XML_PATH, i, base_seed) 
            for i in range(n)
        ]

        for future in concurrent.futures.as_completed(futures):
            try:
                image_filename, joint_position = future.result()
                joint_positions.append(joint_position)
                filenames.append(image_filename)
                pbar.update(1)
            except Exception as e:
                print(f"An error occurred in a thread: {e}")
                pbar.close()
                return # exit if an error has occurred

    pbar.close()
    print("Processing image data and creating conversations...")

    # Create conversations with tqdm progress
    conversations = []
    for i in tqdm(range(len(filenames)), desc="Creating conversations", unit="conv"):
        # Format each joint angle with the special token format
        joint_description = ""
        for j, name in enumerate(joint_names):
            if j < len(joint_positions[i]):
                angle_value = round(joint_positions[i][j], 4)
                joint_description += f"<{name}>{angle_value}</{name}>"
            
        conversation = [
            {"role": "system", "content": f"{SYSTEM_PROMPT}"},
            {"role": "user", "content": [
                    {
                        "type": "image",
                        "content": f"data/{filenames[i]}"
                    },
                    {
                        "type": "text",
                        "content": "<Pose>"
                    }
                ]
            },
            {"role": "assistant", "content":
                [
                    {
                        "type": "text",
                        "content": f"{joint_description}"
                    },
                ]
            },
        ]
        conversations.append(conversation)

    conversations_json = [json.dumps(conv) for conv in conversations]
    output_path = "data/conversations_dataset.jsonl"
    with open(output_path, 'w') as f:
        for conv_json in conversations_json:
            f.write(conv_json + '\n')

    # Open images for dataset creation
    print("Loading images for dataset...")
    for image_path in tqdm(filenames, desc="Loading images", unit="img"):
        try:
            images_data.append(Image.open(f"data/{image_path}"))
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Use a placeholder or skip
            images_data.append(None)

    # Remove any None entries from dataset
    valid_indices = [i for i, img in enumerate(images_data) if img is not None]
    clean_images = [images_data[i] for i in valid_indices]
    clean_conversations = [conversations_json[i] for i in valid_indices]
    
    if len(clean_images) < len(images_data):
        print(f"Warning: {len(images_data) - len(clean_images)} images couldn't be loaded and were skipped")

    dataset_dict = {
        "image": clean_images,
        "conversations": clean_conversations,
    }

    # Create the Hugging Face dataset
    features = Features({
        "image": DsImage(),
        "conversations": Value("string"),
    })

    print("Building dataset object...")
    dataset = Dataset.from_dict(dataset_dict, features=features)

    # Save metadata as JSON
    print("Creating metadata...")
    model_joint_names = [model.joint(i).name for i in range(model.njnt)]
    metadata = {
        "joint_names": model_joint_names,
        "dataset_description": "Random hand poses generated with MuJoCo",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_poses": len(clean_images)
    }

    with open(os.path.join("data", "_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    if push_to_hub:
        try:
            print(f"Pushing dataset to Hugging Face Hub as {dataset_name}")
            # Ensure we have enough samples for test split
            if len(clean_images) <= num_test_sample:
                num_test_sample = max(1, int(len(clean_images) * 0.1))  # 10% for testing
                print(f"Adjusted test sample size to {num_test_sample}")
                
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
                repo_type="dataset"
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
    for img in clean_images:
        try:
            img.close()
        except Exception:
            pass

if __name__ == "__main__":
    import argparse
    from huggingface_hub import HfApi
    from datasets import Dataset

    parser = argparse.ArgumentParser(
        description="Generate n random hand poses and upload to Hugging Face."
    )
    parser.add_argument("n", type=int, help="Number of poses to generate")
    parser.add_argument("--dataset_name", type=str, default="your_username/hand-poses-dataset",
                      help="Name of the Hugging Face dataset to create")
    parser.add_argument("--no_push", action="store_true",
                      help="Don't push to Hugging Face, just save locally")
    parser.add_argument("--num_test_samples", type=int, default="1000",
                        help="Number of test samples to split")
    parser.add_argument("--num_process", type=int, default=None,
                        help="Number of thread to execute task")

    args = parser.parse_args()

    print(f"Starting generation of {args.n} hand poses...")
    start_time = time.time()
    get_n_pose_and_upload(args.n, args.dataset_name, not args.no_push, args.num_test_samples, args.num_process)
    end_time = time.time()

    elapsed_time = end_time - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    print(f"Time elapsed: {int(hours)}h {int(minutes)}m {seconds:.2f}s")