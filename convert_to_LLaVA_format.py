import json


def convert_jsonl_to_vlm_format(input_jsonl_path, output_json_path):
    """
    Convert multi-line JSONL file to a single VLM training format JSON file.

    :param input_jsonl_path: Path to the input JSONL file
    :param output_json_path: Path to save the output JSON file
    """
    # List to store all converted entries
    all_data = []

    # Counter for processed entries
    entry_counter = 0

    # Read the JSONL file
    with open(input_jsonl_path, "r") as jsonl_file:
        for line in jsonl_file:
            try:
                # Parse each line of the JSONL file
                jsonl_data = json.loads(line.strip())

                # Extract image path
                image_path = next(
                    item["content"]
                    for item in jsonl_data[1]["content"]
                    if item["type"] == "image"
                )

                # Extract system prompt
                system_prompt = jsonl_data[0]["content"]

                # Extract messages
                messages = [
                    {"content": system_prompt, "role": "system"},
                    {"content": "<image><Pose>", "role": "user"},
                    {
                        "content": jsonl_data[2]["content"][0]["content"],
                        "role": "assistant",
                    },
                ]

                # Create the output format for this entry
                entry_data = {"messages": messages, "images": [image_path]}

                # Append to the list of all data
                all_data.append(entry_data)

                # Increment entry counter
                entry_counter += 1

                print(f"Processed entry {entry_counter}")

            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line: {line}")
            except Exception as e:
                print(f"Error processing line: {e}")

    # Write all data to a single JSON file
    with open(output_json_path, "w") as json_file:
        json.dump(all_data, json_file, indent=2)

    print(f"\nTotal entries converted: {entry_counter}")
    print(f"All data saved to {output_json_path}")


# Example usage
input_jsonl_path = (
    "./data/conversations_dataset.jsonl"  # Replace with your input JSONL file path
)
output_json_path = (
    "robot_hand_pose.json"  # Replace with your desired output JSON file path
)

convert_jsonl_to_vlm_format(input_jsonl_path, output_json_path)
