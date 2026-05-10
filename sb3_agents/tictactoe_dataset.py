import json
import os
from PIL import Image as PILImage, ImageDraw, ImageFont
from datasets import Dataset, Features, Image as HFImage, Sequence, Value


# TicTacToe dataset
with open(
        "/mnt/data/home/makuke637/SB3_Agents/tictactoe/tictactoe.json",
        "r",
) as f:
    tictactoe = json.load(f)

env_name = tictactoe["name"]


def render_tictactoe_board_to_image(board_state, width, height):
    # Create a white image
    img = PILImage.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)

    # Try to load a nice font, fall back to default if not available
    font = ImageFont.truetype("/usr/share/fonts/gnu-free/FreeMono.ttf", 20)

    # Render the board text
    text = '\n'.join(board_state)

    # Calculate text position to center it
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    x = (width - text_width) // 2
    y = (height - text_height) // 2

    draw.text((x, y), text, fill='black', font=font)

    return img


def dataset_generator(shards):
    for shard in shards:
        sample_group = tictactoe["samples"][shard]
        print(f"Generating dataset for sample {sample_group['sample_id']}")

        for sample in sample_group["rollout"]:
            example = {
                "messages": {
                    "name": env_name,
                    "state": '\n'.join(sample["state"]),
                    "action": sample["action"],
                    "reward": sample["reward"]["X"],
                    "score": sample["score"]["X"],
                    "lives": None,
                    "terminated": sample["status"]["terminated"],
                    "truncated": sample["status"]["truncated"],
                    "started": sample["status"]["started"],
                    "reasoning": sample["reasoning"],
                    "step": sample["step"],
                    "img_embed": None,
                },
                "images": [render_tictactoe_board_to_image(
                    sample["state"], width=200, height=200
                )],
            }
            # print(example)

            yield example


# load datasets from folder
shards = list(range(len(tictactoe["samples"])))
cpus = os.cpu_count()
dataset = Dataset.from_generator(
    dataset_generator,
    features=Features(
        {
            "messages": {
                "name": Value("string"),
                "state": Value("string"),
                "action": Value("string"),
                "reward": Value("float32"),
                "score": Value("float32"),
                "lives": Value("int64"),
                "terminated": Value("bool"),
                "truncated": Value("bool"),
                "started": Value("bool"),
                "reasoning": Value("string"),
                "step": Value("int64"),
                "img_embed": Sequence(Sequence(Value("float32"))),
            },
            "images": Sequence(HFImage()),
        }
    ),
    num_proc=cpus,
    gen_kwargs={"shards": shards},
)
print("Total samples:", len(dataset))

# Save the dataset
ds_path = "/mnt/data/home/makuke637/SB3_Agents/dataset"
os.makedirs(ds_path, exist_ok=True)
dataset.save_to_disk(
    os.path.join(ds_path, f"{env_name}"),
    num_proc=cpus,
)