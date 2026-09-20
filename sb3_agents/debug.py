import datetime
import cv2
import numpy as np
from datasets import load_from_disk

### Environments
env_names = (
    ### Atari 2600
    # "AssaultNoFrameskip-v4",
    # "AtlantisNoFrameskip-v4",
    # "BankHeistNoFrameskip-v4",
    # "BoxingNoFrameskip-v4",
    # "BreakoutNoFrameskip-v4",
    # "CrazyClimberNoFrameskip-v4",
    # "DefenderNoFrameskip-v4",
    # "DemonAttackNoFrameskip-v4",
    # "DoubleDunkNoFrameskip-v4",
    # "EnduroNoFrameskip-v4",
    # "FishingDerbyNoFrameskip-v4",
    # "FreewayNoFrameskip-v4",
    # "GopherNoFrameskip-v4",
    # "JamesbondNoFrameskip-v4",
    # "KangarooNoFrameskip-v4",
    # "KrullNoFrameskip-v4",
    # "KungFuMasterNoFrameskip-v4",
    # "PhoenixNoFrameskip-v4",
    # "PongNoFrameskip-v4",
    # "QbertNoFrameskip-v4",
    # "RoadRunnerNoFrameskip-v4",
    # "StarGunnerNoFrameskip-v4",
    # "TutankhamNoFrameskip-v4",
    # "UpNDownNoFrameskip-v4",
    # "VideoPinballNoFrameskip-v4",

    ### Classic Environments
    "Pendulum-v1",
    "CartPole-v1",
    "MountainCar-v0",
    "Acrobot-v1",
    "LunarLander-v3",
)


for env_name in env_names:
    # Load the dataset from disk
    ds = load_from_disk(f"./dataset/{env_name}")

    # Create a video writer object to save the frames as a video
    video_out = cv2.VideoWriter(
        filename=f"./videos/{env_name}_{datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}.mp4",
        fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
        fps=25,
        frameSize=(1000, 1000),
    )

    for row in ds:
        for frame_id, img in enumerate(row['images']):
            # Convert to BGR format for OpenCV
            frame = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

            # Resize the frame
            frame = cv2.resize(frame, (1000, 1000), interpolation=cv2.INTER_LANCZOS4)

            # Add text to the frame
            cv2.putText(
                frame,
                f"Game: {row['messages']['game']}",
                (40, 70),
                cv2.FONT_HERSHEY_DUPLEX,
                0.8,
                (234, 232, 233),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"Agent: {row['messages']['name']}",
                (40, 100),
                cv2.FONT_HERSHEY_DUPLEX,
                0.8,
                (234, 232, 233),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"Action: {row['messages']['action']}",
                (40, 130),
                cv2.FONT_HERSHEY_DUPLEX,
                0.8,
                (234, 232, 233),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"Reward: {row['messages']['reward']}",
                (40, 160),
                cv2.FONT_HERSHEY_DUPLEX,
                0.8,
                (234, 232, 233),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"Started: {row['messages']['started']}",
                (40, 190),
                cv2.FONT_HERSHEY_DUPLEX,
                0.8,
                (234, 232, 233),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"Termination: {row['messages']['termination']}",
                (40, 220),
                cv2.FONT_HERSHEY_DUPLEX,
                0.8,
                (234, 232, 233),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"Truncation: {row['messages']['truncation']}",
                (40, 250),
                cv2.FONT_HERSHEY_DUPLEX,
                0.8,
                (234, 232, 233),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"Draw: {row['messages']['draw']}",
                (40, 280),
                cv2.FONT_HERSHEY_DUPLEX,
                0.8,
                (234, 232, 233),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"Lives: {row['messages']['lives']}",
                (40, 310),
                cv2.FONT_HERSHEY_DUPLEX,
                0.8,
                (234, 232, 233),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"FrameID: {frame_id}",
                (40, 340),
                cv2.FONT_HERSHEY_DUPLEX,
                0.8,
                (234, 232, 233),
                2,
                cv2.LINE_AA,
            )

            # Write the frame to the video
            video_out.write(frame)

    video_out.release()