# Poseless: Direct Hand-to-Robot Joint Angle Estimation

Poseless is a project that enables direct estimation of robot hand joint angles from human hand input without requiring complex 3D skeleton structure or pose estimation pipelines. This simplified approach provides a more efficient way to map human hand movements to robotic hand control.

## Overview

The system directly maps human hand input to robot joint angles, bypassing the need for intermediate 3D pose estimation. This approach offers:
- Reduced computational complexity
- Lower latency
- Direct joint angle mapping
- Simplified implementation

## Getting Started

### Prerequisites
- Python 3.x
- Required dependencies please check `requirements.txt`

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/poseless.git
cd poseless

# Install dependencies (to be added)
sudo apt update                                                                                                                                    
sudo apt install libegl1 libegl-dev
ldconfig -p | grep libEGL #check if 
pip install -r requirements.txt
```

## Usage
To generate samples, use the following command:
```bash
export PYOPENGL_PLATFORM=egl
export MUJOCO_GL=egl  # Also set this.
python gen_multithread.py <number_to_generate> --dataset_name="your-username/hand-poses-dataset" --num_test_samples <num_test_samples>
```

