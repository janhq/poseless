# %%
import mujoco
from PIL import Image
import numpy as np
import json
import os

model = mujoco.MjModel.from_xml_path("robotics-models/shadow_hand/left_hand.xml")
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, height=1280, width=1280)


# Get joint information and set random positions
def get_n_pose(n):
    os.makedirs("data", exist_ok=True)
    poses_data = {}
    pose_num = 0

    while len(poses_data) < n:
        mujoco.mj_resetData(model, data)

        target_positions = []

        for i in range(2, model.njnt):
            joint = model.joint(i)
            # Generate random value within range
            target_pos = np.random.uniform(joint.range[0], joint.range[1])
            target_positions.append((joint.qposadr[0], target_pos))

            # Set joint value
            # qpos_addr = joint.qposadr[0]
            # data.qpos[qpos_addr] = joint_pos

        for step in range(2000):
            # Apply control to move toward target positions
            for qpos_addr, target_pos in target_positions:
                current_pos = data.qpos[qpos_addr]
                # Simple PD control (proportional-derivative)
                kp = 10.0  # Proportional gain
                error = target_pos - current_pos
                # Apply torque/force through qfrc_applied
                data.qfrc_applied[qpos_addr] = kp * error

            # Step physics (this will respect collisions)
            mujoco.mj_step(model, data)

            # Check if we're close enough to targets or if we've reached a stable state
            if step % 100 == 0:  # Check periodically to save computation
                total_error = 0
                for qpos_addr, target_pos in target_positions:
                    total_error += abs(target_pos - data.qpos[qpos_addr])

                # If close enough to target or hand has stabilized with collisions
                if total_error < 0.1 or (step > 500 and data.ncon < 10):
                    break

        # for _ in range(1000):
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera="closeup")

        # mujoco.mj_forward(model, data)
        # check_collisions()

        # if data.ncon > 17:
        #     continue

        # Update the scene and render
        pixels = renderer.render()

        image_filename = f"pose_{pose_num}.png"
        image_path = os.path.join("data", image_filename)
        image = Image.fromarray(pixels)
        image.save(image_path)

        # Store the qpos data for this pose
        poses_data[image_filename] = data.qpos.copy().tolist()

        # print(f"{image_filename}: {poses_data[image_filename]}")

        pose_num += 1

    # Save all poses data to a JSON file

    json_path = os.path.join("data", "_index.json")
    with open(json_path, "w") as f:
        json.dump(poses_data, f, indent=2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate n random hand poses and save their images and joint positions."
    )
    parser.add_argument("n", type=int, help="Number of poses to generate")

    args = parser.parse_args()
    get_n_pose(args.n)
