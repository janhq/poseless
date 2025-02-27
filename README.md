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
pip install -r requirements.txt
```

## Usage
To generate samples, use the following command:
```bash
python gen.py <n>
```
where `<n>` is the number of samples you want to generate.

### Data Format
The generated data is stored in the `data` folder with the following structure:
- `_index.json`: Contains the joint angle values where the keys are the image file's name
- `_name.json`: Contains the corresponding joint names for each angle value
- `pose_{n}.png`: Contains the image corresponding with the angle values in `_index`

Example:
```
data/
  ├── _index.json
  ├── _name.json
  └── pose_0.png
```

## Project Structure
```
poseless/
├── data/               # Generated sample data
├── shadow_hand/        # The model for shadowhand dexterous hand 
├── gen.py              # Sample generation script
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation
```
