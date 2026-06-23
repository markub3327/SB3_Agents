import numpy as np

from utils import list_leaf_dirs
from datasets import interleave_datasets, load_from_disk
import cv2
import os


# Load pre-processed dataset
dataset_paths = list_leaf_dirs("/mnt/data/home/makuke637/SB3_Agents/dataset/")
print(f"Dataset paths: {dataset_paths}")
dataset = interleave_datasets(
    datasets=[load_from_disk(path) for path in dataset_paths],
    stopping_strategy="all_exhausted",
)
print(f"Dataset examples: {len(dataset)}")

for path in dataset_paths:
    ds = load_from_disk(path)
    print(f"{path}:")
    print(f"  - Examples: {len(ds)}")
    print(f"  - Columns: {ds.column_names}")
    print(f"  - Features: {ds.features}")
    print()

video_out = cv2.VideoWriter(
    os.path.join("./videos/", f"result.mp4"),
    cv2.VideoWriter_fourcc(*"mp4v"),
    30,
    (400, 400),
)

for i in range(0, len(dataset), 7):
    data = dataset[i]
    print(f"Processing sample {i+1}...")

    for j, img in enumerate(np.asarray(data["images"])):
        print(img.shape)
        frame = cv2.resize(img, (400, 400))

        # Add text to the frame
        cv2.putText(
            frame,
            f"Action: {data['messages']['action']}",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            (112, 128, 144),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Reward: {data['messages']['reward']}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            (112, 128, 144),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Started: {data['messages']['started']}",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            (112, 128, 144),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Terminated: {data['messages']['done']}",
            (10, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            (112, 128, 144),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"State: {data['messages']['state']}",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            (112, 128, 144),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Reasoning: {data['messages']['reasoning']}",
            (10, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            (112, 128, 144),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Frame ID: {j}",
            (10, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            (112, 128, 144),
            1,
            cv2.LINE_AA,
        )
        video_out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


video_out.release()

