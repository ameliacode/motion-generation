TARGET_FPS = 30
WINDOW_SIZE = 160
CHANNEL_SIZE = 63
FILTER_SIZE = 15
OVERLAP = 80
NUM_JOINTS = 20

ALPHA = 0.01
EPOCHS = 25
BATCH_SIZE = 32

JOINTS = [
    "Hips",
    "LowerBack",
    "Spine",
    "Spine1",
    "Neck",
    "Head",
    "RightShoulder",
    "RightArm",
    "RightForeArm",
    "RightHand",
    "LeftShoulder",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
    "RightUpLeg",
    "RightLeg",
    "RightFoot",
    "LeftUpLeg",
    "LeftLeg",
    "LeftFoot",
]

BONES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 5),
    (3, 6),
    (6, 7),
    (7, 8),
    (8, 9),
    (3, 10),
    (10, 11),
    (11, 12),
    (12, 13),
    (0, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
]
