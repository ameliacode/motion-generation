import numpy as np
from config import *
from keras import optimizers
from model import *
from tqdm.keras import TqdmCallback

data = np.load("./data/03_data.npz")
X = data["data"].astype(np.float32)
end_indices = data["end_indices"].astype(np.int32)


def preprocess(pose_data, end_indices):
    prev_poses = []
    curr_poses = []

    start = 0
    for end in end_indices:
        for i in range(start, end):
            prev_poses.append(pose_data[i])
            curr_poses.append(pose_data[i + 1])
        start = end + 1

    return np.array(prev_poses), np.array(curr_poses)


prev_poses, curr_poses = preprocess(X, end_indices)
print(f"Training pairs: {len(prev_poses)}")

callbacks_list = [
    TqdmCallback(verbose=1),
]

cvae.compile(optimizer=optimizers.Adam(learning_rate=LEARNING_RATE))

cvae.fit(
    [prev_poses, curr_poses],
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks_list,
    verbose=0,
)


print("Training done! Run: tensorboard --logdir logs")
