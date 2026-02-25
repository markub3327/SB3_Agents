from bidict import bidict

full_action_space = bidict(
    {
        "noop": 0,
        "fire": 1,
        "up": 2,
        "right": 3,
        "left": 4,
        "down": 5,
        "upright": 6,
        "upleft": 7,
        "downright": 8,
        "downleft": 9,
        "upfire": 10,
        "rightfire": 11,
        "leftfire": 12,
        "downfire": 13,
        "uprightfire": 14,
        "upleftfire": 15,
        "downrightfire": 16,
        "downleftfire": 17,
        "reset": 40,
    }
)
ids_action_vocab = {
    "GopherNoFrameskip-v4": bidict(
        {
            "noop": 0,
            "fire": 1,
            "up": 2,
            "right": 3,
            "left": 4,
            "upfire": 5,
            "rightfire": 6,
            "leftfire": 7,
        }
    ),
    "NameThisGameNoFrameskip-v4": bidict(
        {
            "noop": 0,
            "fire": 1,
            "right": 2,
            "left": 3,
            "rightfire": 4,
            "leftfire": 5,
        }
    ),
    "RoadRunnerNoFrameskip-v4": full_action_space,
    "QbertNoFrameskip-v4": bidict(
        {
            "noop": 0,
            "fire": 1,
            "up": 2,
            "right": 3,
            "left": 4,
            "down": 5,
        }
    ),
    "AssaultNoFrameskip-v4": bidict(
        {
            "noop": 0,
            "fire": 1,
            "up": 2,
            "right": 3,
            "left": 4,
            "rightfire": 5,
            "leftfire": 6,
        }
    ),
    "BreakoutNoFrameskip-v4": bidict(
        {
            "noop": 0,
            "fire": 1,
            "right": 2,
            "left": 3,
        }
    ),
    "FreewayNoFrameskip-v4": bidict(
        {
            "noop": 0,
            "up": 1,
            "down": 2,
        }
    ),
    "StarGunnerNoFrameskip-v4": full_action_space,
    "PongNoFrameskip-v4": bidict(
        {
            "noop": 0,
            "fire": 1,
            "right": 2,
            "left": 3,
            "rightfire": 4,
            "leftfire": 5,
        }
    ),
    "BoxingNoFrameskip-v4": full_action_space,
    "FishingDerbyNoFrameskip-v4": full_action_space,
    "AtlantisNoFrameskip-v4": bidict(
        {
            "noop": 0,
            "fire": 1,
            "rightfire": 2,
            "leftfire": 3
        }
    ),
    "CrazyClimberNoFrameskip-v4": bidict(
        {
            "noop": 0,
            "up": 1,
            "right": 2,
            "left": 3,
            "down": 4,
            "upright": 5,
            "upleft": 6,
            "downright": 7,
            "downleft": 8
        }
    ),
    "DefenderNoFrameskip-v4": full_action_space,
    "EnduroNoFrameskip-v4": bidict(
        {
            "noop": 0,
            "fire": 1,
            "right": 2,
            "left": 3,
            "down": 4,
            "downright": 5,
            "downleft": 6,
            "rightfire": 7,
            "leftfire": 8,
        }
    ),
    "JamesbondNoFrameskip-v4": full_action_space,
    "PhoenixNoFrameskip-v4": bidict(
        {
            "noop": 0,
            "fire": 1,
            "right": 2,
            "left": 3,
            "down": 4,
            "rightfire": 5,
            "leftfire": 6,
            "downfire": 7,
        }
    )
}
