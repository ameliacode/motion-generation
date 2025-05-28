import glob
import os
from warnings import simplefilter

import numpy as np
from sklearn.pipeline import Pipeline

from pymo.parsers import BVHParser
from pymo.preprocessing import *
from pymo.viz_tools import *

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


def normalize_data(data):
    """Normalize data to unit variance"""
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    std[std == 0] = 1  # Avoid division by zero
    return (data - mean) / std


def process_bvh_file(filepath):
    parser = BVHParser()
    parsed_data = parser.parse(filepath)

    data_pipe = Pipeline(
        [
            ("param", MocapParameterizer("position")),
            ("jtsel", JointSelector(JOINTS, include_root=False)),
            # ('jtnorm', JointLengthNormalizer()),
            ("dwnsampl", DownSampler(tgt_fps=30, keep_all=False)),
            ("globrm", GlobalMotionRemover()),
            ("np", Numpyfier()),
        ]
    )
    piped_data = data_pipe.fit_transform([parsed_data])
    slicer = Slicer(window_size=160, overlap=0.5)
    piped_data = slicer.fit_transform(piped_data)
    return piped_data


def main():
    bvh_files = glob.glob(os.path.join(BVH_DIR, "*.bvh"))
    bvh_files.extend(glob.glob(os.path.join(BVH_DIR, "**/*.bvh"), recursive=True))

    all_windows = []
    file_names = []

    for bvh_file in bvh_files:
        windows = process_bvh_file(bvh_file)
        print(windows.shape)
        if windows is not None and windows.shape[0] != 0:
            all_windows.append(windows)
            file_names.append(os.path.basename(bvh_file))

    final_data = np.concatenate(all_windows, axis=0)
    final_mean = np.mean(final_data, axis=0)
    final_std = np.std(final_data, axis=0)

    np.savez_compressed(OUTPUT_FILE, clips=final_data, mean=final_mean, std=final_std)

    print(final_data.shape)


if __name__ == "__main__":
    main()
