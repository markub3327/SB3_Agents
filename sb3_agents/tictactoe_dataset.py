import json
import os
import numpy as np
from PIL import Image as PILImage, ImageDraw, ImageFont
from datasets import Dataset, Features, Image as HFImage, Sequence, Value
import cv2


# TicTacToe dataset
with open(
        "/mnt/data/home/makuke637/SB3_Agents/tictactoe/tictactoe.json",
        "r",
) as f:
    tictactoe = json.load(f)


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


def dataset_wrapper(game):
    def dataset_generator(shards):
        for shard in shards:
            sample_group = game["samples"][shard]
            print(f"Generating dataset for sample {sample_group['sample_id']}")

            for sample in sample_group["rollout"]:
                example = {
                    "messages": {
                        "name": env_name,
                        "state": '\n'.join(sample["state"]),
                        "action": sample["action"],
                        "reward": sample["reward"],
                        "done": sample["status"]["terminated"],
                        "started": sample["status"]["started"],
                        "reasoning": sample["reasoning"],
                    },
                    "images": sample["img"],
                }
                # print(example)

                yield example

    return dataset_generator


for game in tictactoe["games"]:
    env_name = game["name"]
    print(env_name)

    # Create states for all samples
    width, height = 400, 400
    for sample_group in game["samples"]:
        for sample in sample_group["rollout"]:
            img = render_tictactoe_board_to_image(sample["state"], width=width, height=height)
            sample["img"] = np.asarray([img], dtype=np.uint8)

    # load datasets from folder
    shards = list(range(len(game["samples"])))
    cpus = os.cpu_count()
    dataset = Dataset.from_generator(
        dataset_wrapper(game),
        features=Features(
            {
                "messages": {
                    "name": Value("string"),
                    "state": Value("string"),
                    "action": Value("string"),
                    "reward": Value("float32"),
                    "done": Value("bool"),
                    "started": Value("bool"),
                    "reasoning": Value("string"),
                },
                "images": Sequence(HFImage()),
            }
        ),
        num_proc=cpus,
        gen_kwargs={"shards": shards},
    )
    print("Total samples:", len(dataset))

    # Shuffle the dataset once before saving
    dataset = dataset.shuffle(seed=42)
    print(f"Dataset shuffled with seed 42")

    # Save the dataset
    ds_path = "/mnt/data/home/makuke637/SB3_Agents/dataset"
    os.makedirs(ds_path, exist_ok=True)
    dataset.save_to_disk(
        os.path.join(ds_path, f"{env_name}"),
        num_proc=cpus,
    )

    os.makedirs("./videos/", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(
        f"./videos/{env_name}.mp4", fourcc, 10, (width, height)
    )
    for samples in game["samples"]:
        for sample in samples["rollout"]:
            # Convert RGB to BGR for OpenCV
            bgr_frame = cv2.cvtColor(sample["img"][0], cv2.COLOR_RGB2BGR)

            action = sample["action"]
            reward = sample["reward"]
            started = sample["status"]["started"]
            terminated = sample["status"]["terminated"]

            # Add text to the frame
            cv2.putText(
                bgr_frame,
                f"Action: {action}",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (112, 128, 144),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                bgr_frame,
                f"Reward: {reward}",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (112, 128, 144),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                bgr_frame,
                f"Started: {started}",
                (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (112, 128, 144),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                bgr_frame,
                f"Terminated: {terminated}",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (112, 128, 144),
                1,
                cv2.LINE_AA,
            )

            video.write(bgr_frame)
    video.release()
    print("Video recorded.")