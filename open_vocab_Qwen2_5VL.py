import torch
from huggingface_hub import HfApi, create_repo
from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration

# we dont have to resize embedding and lm_head cause the number of padding is 271.
device = torch.device("cpu")
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

# Load the model and tokenizer
model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype="auto", device_map="cpu"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
old_vocab_size = len(tokenizer)
print(old_vocab_size)

task_token = ["<Pose>"]
joint_token = []
for name in joint_names:
    joint_token.append(f"<{name}>")
    joint_token.append(f"</{name}>")
add_tokens = task_token + joint_token
print(len(add_tokens))

# Add new vocab
tokenizer.add_tokens(add_tokens)
output_dir = "../Qwen2.5-VL-3B-Instruct-Resized/"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

# Push to HF hub
repo_id = "jan-hq/Qwen2.5-VL-3B-Instruct-Resized"
api = HfApi()
create_repo(repo_id, exist_ok=True)
api.upload_folder(
    folder_path=f"{output_dir}",
    repo_id=f"{repo_id}",
    repo_type="model",
)
print("Model and tokenizer updated and pushed to Hugging Face Hub.")
