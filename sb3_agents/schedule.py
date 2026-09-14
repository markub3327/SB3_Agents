import math


class CosineAnnealingLR:
    def __init__(self, learning_rate):
        self.learning_rate = float(learning_rate)

    def __call__(self, progress_remaining):
        return (
            self.learning_rate
            * 0.5
            * (1.0 + math.cos(math.pi * (1.0 - progress_remaining)))
        )

    def __repr__(self):
        return f"CosineAnnealingLR(learning_rate={self.learning_rate})"