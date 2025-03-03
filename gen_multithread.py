import concurrent.futures
import copy
import io
import json
import os
import time
from multiprocessing import cpu_count

import mujoco
import numpy as np
from datasets import Dataset, Features
from datasets import Image as DsImage
from datasets import Value
from huggingface_hub import HfApi, login
from PIL import Image
from tqdm.auto import tqdm

os.environ["MUJOCO_GL"] = "egl"

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

nthread = 32


def calculate_thread_ranges(total_poses, num_threads):
    """
    Calculate start and end indices for each thread to evenly distribute poses

    Args:
        total_poses (int): Total number of poses to generate
        num_threads (int): Number of threads to use

    Returns:
        List of (start_index, end_index) tuples for each thread
    """
    base_chunk_size = total_poses // num_threads
    remainder = total_poses % num_threads

    # Prepare thread ranges
    thread_ranges = []
    current_start = 0

    for thread in range(num_threads):

        current_chunk_size = base_chunk_size + (1 if thread < remainder else 0)
        current_end = current_start + current_chunk_size
        thread_ranges.append((current_start, current_end))
        current_start = current_end

    return thread_ranges


def generate_pose_chunk(thread_id, start_idx, end_idx):
    """Generate a chunk of poses using MuJoCo simulation"""
    # Create local model and data copies for this thread
    local_model = copy.deepcopy(model)
    local_data = mujoco.MjData(local_model)
    # Create a local renderer for this thread
    local_renderer = mujoco.Renderer(local_model, height=1280, width=1280)

    chunk_filenames = []
    chunk_joint_positions = []

    # Generate poses for this thread's range
    for pose_num in range(start_idx, end_idx):
        mujoco.mj_resetData(local_model, local_data)

        # Generate random joint positions
        target_positions = []
        for j in range(2, local_model.njnt):
            joint = local_model.joint(j)
            target_pos = np.random.uniform(joint.range[0], joint.range[1])
            target_positions.append((joint.qposadr[0], target_pos))

        # Apply control to reach target positions using MuJoCo's step function
        for step in range(2000):
            for qpos_addr, target_pos in target_positions:
                current_pos = local_data.qpos[qpos_addr]
                kp = 10.0  # Proportional gain
                error = target_pos - current_pos
                local_data.qfrc_applied[qpos_addr] = kp * error

            # Step physics
            mujoco.mj_step(local_model, local_data)

            # Check if close enough to targets
            if step % 100 == 0:
                total_error = 0
                for qpos_addr, target_pos in target_positions:
                    total_error += abs(target_pos - local_data.qpos[qpos_addr])
                if total_error < 0.1 or (step > 500 and local_data.ncon < 10):
                    break

        # Forward kinematics and render
        mujoco.mj_forward(local_model, local_data)
        local_renderer.update_scene(local_data, camera="closeup")
        pixels = local_renderer.render()

        image_filename = f"pose_{pose_num:06d}.png"
        image_path = os.path.join("data", image_filename)
        image = Image.fromarray(pixels)
        image.save(image_path)

        chunk_joint_positions.append(local_data.qpos.copy().tolist())
        chunk_filenames.append(image_filename)

    return chunk_filenames, chunk_joint_positions


def get_n_pose_and_upload(
    n, dataset_name="hand-poses-dataset", push_to_hub=True, num_test_sample=1000
):
    """Generate n random hand poses using parallel processing and upload to Hugging Face."""
    global joint_names, joint_name_to_index, SYSTEM_PROMPT, nthread

    assert (
        num_test_sample < n
    ), "The number of test samples must be lower than the total synthetic subset"

    os.makedirs("data", exist_ok=True)

    thread_ranges = calculate_thread_ranges(n, nthread)

    all_filenames = []
    all_joint_positions = []

    print(f"Using {nthread} threads to generate {n} poses")
    print("Thread ranges:", thread_ranges)

    # Multi-threaded pose generation
    with concurrent.futures.ThreadPoolExecutor(max_workers=nthread) as executor:
        futures = []
        for thread_id, (start_idx, end_idx) in enumerate(thread_ranges):
            futures.append(
                executor.submit(
                    generate_pose_chunk,
                    thread_id,  # thread identifier
                    start_idx,  # start index for this thread
                    end_idx,  # end index for this thread
                )
            )

        # Collect results with progress bar
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=nthread,
            desc="Generating pose chunks",
        ):
            chunk_filenames, chunk_joint_positions = future.result()
            all_filenames.extend(chunk_filenames)
            all_joint_positions.extend(chunk_joint_positions)

    print("Processing image data and creating conversations...")

    # Create conversations with progress tracking
    conversations = []
    for i in tqdm(
        range(len(all_filenames)), desc="Creating conversations", unit="conv"
    ):
        # Format joint angles with special token format
        joint_description = ""
        for name in joint_names:
            if name in joint_name_to_index:
                joint_idx = joint_name_to_index[name]
                if joint_idx < len(all_joint_positions[i]):
                    angle_value = round(all_joint_positions[i][joint_idx], 4)
                    joint_description += f"<{name}>{angle_value}</{name}>"

        conversation = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image", "content": f"data/{all_filenames[i]}"},
                    {"type": "text", "content": "<Pose>"},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "content": joint_description},
                ],
            },
        ]
        conversations.append(conversation)

    # Save conversations to JSONL file
    conversations_json = [json.dumps(conv) for conv in conversations]
    output_path = "data/conversations_dataset.jsonl"
    with open(output_path, "w") as f:
        for conv_json in conversations_json:
            f.write(conv_json + "\n")

    # Load images for dataset creation
    print("Loading images for dataset...")
    images_data = []
    for image_path in tqdm(all_filenames, desc="Loading images"):
        images_data.append(Image.open(f"data/{image_path}"))

    # Create dataset dictionary
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

    # Save metadata
    print("Creating metadata...")
    model_joint_names = [model.joint(i).name for i in range(model.njnt)]
    metadata = {
        "joint_names": model_joint_names,
        "dataset_description": "Random hand poses generated with MuJoCo",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_poses": n,
        "generation_method": "GPU-accelerated MuJoCo with multi-threading",
    }

    with open(os.path.join("data", "_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # Push to Hugging Face Hub or save locally
    if push_to_hub:
        try:
            print(f"Pushing dataset to Hugging Face Hub as {dataset_name}")
            dataset = dataset.train_test_split(test_size=num_test_sample)
            dataset.push_to_hub(dataset_name)

            # Push metadata
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
        dataset.save_to_disk("data/hf_dataset")

    # Clean up resources
    print("Cleaning up resources...")
    for img in images_data:
        try:
            img.close()
        except:
            pass

    return dataset


# Use MuJoCo's rollout for batch simulation (alternative approach)
def get_n_pose_with_rollout(
    n, dataset_name="hand-poses-dataset", push_to_hub=True, num_test_sample=1000
):
    global joint_names, joint_name_to_index, SYSTEM_PROMPT, nthread
    """Generate n random hand poses using MuJoCo's rollout for batch simulation."""
    try:
        from mujoco import rollout

        use_rollout = True
    except ImportError:
        print(
            "MuJoCo rollout module not available, falling back to standard implementation"
        )
        return get_n_pose_and_upload(n, dataset_name, push_to_hub, num_test_sample)

    print("Using MuJoCo rollout module for batch simulation")
    os.makedirs("data", exist_ok=True)

    # Create multiple data instances for threading
    datas = [copy.copy(data) for _ in range(nthread)]

    # Batch parameters
    batch_size = min(1000, n)  # Process in batches to avoid memory issues
    num_batches = (n + batch_size - 1) // batch_size

    all_filenames = []
    all_joint_positions = []

    for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
        current_batch_size = min(batch_size, n - batch_idx * batch_size)

        # Initial states with random positions
        initial_states = []
        for _ in range(current_batch_size):
            # Reset data
            mujoco.mj_resetData(model, data)

            # Set random target positions
            for j in range(2, model.njnt):
                joint = model.joint(j)
                data.qpos[joint.qposadr[0]] = np.random.uniform(
                    joint.range[0], joint.range[1]
                )

            # Get state
            state = np.zeros(
                (mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS),)
            )
            mujoco.mj_getState(model, data, state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
            initial_states.append(state)

        initial_states = np.array(initial_states)

        # Use rollout to simulate all poses in parallel
        # Run a short simulation to settle the hand
        nstep = 2000  # Adjust as needed
        states, _ = rollout.rollout(model, datas, initial_states, nstep=nstep)

        # Process the final states from the rollout
        batch_filenames = []
        batch_joint_positions = []

        # Render and save each pose
        for i in range(current_batch_size):
            pose_num = batch_idx * batch_size + i

            # Set the state to the final simulation state
            mujoco.mj_setState(
                model, data, states[i, -1, :], mujoco.mjtState.mjSTATE_FULLPHYSICS
            )
            mujoco.mj_forward(model, data)

            # Render the pose
            renderer.update_scene(data, camera="closeup")
            pixels = renderer.render()

            # Save the image
            image_filename = f"pose_{pose_num}.png"
            image_path = os.path.join("data", image_filename)
            image = Image.fromarray(pixels)
            image.save(image_path)

            # Store results
            batch_joint_positions.append(data.qpos.copy().tolist())
            batch_filenames.append(image_filename)

        all_filenames.extend(batch_filenames)
        all_joint_positions.extend(batch_joint_positions)

    print("Processing image data and creating conversations...")

    # Create conversations
    conversations = []
    for i in tqdm(
        range(len(all_filenames)), desc="Creating conversations", unit="conv"
    ):
        joint_description = ""
        for name in joint_names:
            if name in joint_name_to_index:
                joint_idx = joint_name_to_index[name]
                if joint_idx < len(all_joint_positions[i]):
                    angle_value = round(all_joint_positions[i][joint_idx], 4)
                    joint_description += f"<{name}>{angle_value}</{name}>"

        conversation = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image", "content": f"data/{all_filenames[i]}"},
                    {"type": "text", "content": "<Pose>"},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "content": joint_description},
                ],
            },
        ]
        conversations.append(conversation)

    # Save to JSONL
    conversations_json = [json.dumps(conv) for conv in conversations]
    output_path = "data/conversations_dataset.jsonl"
    with open(output_path, "w") as f:
        for conv_json in conversations_json:
            f.write(conv_json + "\n")

    # Load images
    images_data = []
    for image_path in tqdm(all_filenames, desc="Loading images"):
        images_data.append(Image.open(f"data/{image_path}"))

    dataset_dict = {
        "image": images_data,
        "conversations": conversations_json,
    }

    features = Features(
        {
            "image": DsImage(),
            "conversations": Value("string"),
        }
    )

    print("Building dataset object...")
    dataset = Dataset.from_dict(dataset_dict, features=features)

    # Save metadata
    model_joint_names = [model.joint(i).name for i in range(model.njnt)]
    metadata = {
        "joint_names": model_joint_names,
        "dataset_description": "Random hand poses generated with MuJoCo",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_poses": n,
        "generation_method": "GPU-accelerated MuJoCo with rollout batch simulation",
    }

    with open(os.path.join("data", "_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # Push to Hugging Face Hub or save locally
    if push_to_hub:
        try:
            print(f"Pushing dataset to Hugging Face Hub as {dataset_name}")
            dataset = dataset.train_test_split(test_size=num_test_sample)
            dataset.push_to_hub(dataset_name)

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
        dataset.save_to_disk("data/hf_dataset")

    # Clean up
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
        default=1000,
        help="Number of test samples to split",
    )
    parser.add_argument(
        "--use_rollout",
        action="store_true",
        help="Use MuJoCo rollout for batch simulation (faster for large batches)",
    )

    args = parser.parse_args()

    print(f"Starting generation of {args.n} hand poses...")
    start_time = time.time()

    if args.use_rollout:
        dataset = get_n_pose_with_rollout(
            args.n, args.dataset_name, not args.no_push, args.num_test_samples
        )
    else:
        dataset = get_n_pose_and_upload(
            args.n, args.dataset_name, not args.no_push, args.num_test_samples
        )

    end_time = time.time()

    elapsed_time = end_time - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    print(f"Time elapsed: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
