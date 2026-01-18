import math


class SimpleLinearSchedule:
    """
    Linear learning rate schedule (from initial value to zero),
    simpler than sb3 LinearSchedule.

    :param initial_value: (float or str) The initial value for the schedule
    """

    def __init__(self, initial_value):
        # Force conversion to float
        self.initial_value = float(initial_value)

    def __call__(self, progress_remaining):
        return progress_remaining * self.initial_value

    def __repr__(self):
        return f"SimpleLinearSchedule(initial_value={self.initial_value})"

class SimpleCosineSchedule:
    """
    Cosine learning rate schedule (from initial value to zero)
    :param initial_value: (float or str) The initial value for the schedule
    """

    def __init__(self, initial_value):
        # Force conversion to float
        self.initial_value = float(initial_value)

    def __call__(self, progress_remaining):
        return self.initial_value * 0.5 * (1.0 + math.cos(math.pi * (1.0 - progress_remaining)))

    def __repr__(self):
        return f"SimpleCosineSchedule(initial_value={self.initial_value})"


def linear_schedule(initial_value):
    """
    Linear learning rate schedule.

    :param initial_value: (float or str)
    :return: A `SimpleLinearSchedule` object
    """
    return SimpleLinearSchedule(initial_value)


def cosine_schedule(initial_value):
    """
    Linear learning rate schedule.

    :param initial_value: (float or str)
    :return: A `SimpleLinearSchedule` object
    """
    return SimpleCosineSchedule(initial_value)