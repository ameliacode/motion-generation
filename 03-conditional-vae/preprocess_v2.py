import glob
import os
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count

import numpy as np
from config import *
from sklearn.pipeline import Pipeline
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.getcwd())))

from pymo.parsers import BVHParser
from pymo.preprocessing import *
from pymo.viz_tools import *

warnings.filterwarnings("ignore")


def get_locomotion_files(data_path="../motionsynth_data/data/processed/cmu"):
    """Get only locomotion category BVH files from CMU dataset."""
    all_files = glob.glob(os.path.join(data_path, "*.bvh"))
    locomotion_files = []

    for file_path in all_files:
        filename = os.path.basename(file_path)
        try:
            subject_num = int(filename.split("_")[0])
            if subject_num in LOCOMOTION_SUBJECTS:
                locomotion_files.append(file_path)
        except (ValueError, IndexError):
            # Skip files that don't follow the expected naming convention
            continue

    return sorted(locomotion_files)


def process_bvh_file(filepath):
    try:
        parser = BVHParser()
        parsed_data = parser.parse(filepath)

        data_pipe = Pipeline(
            [
                ("parameterizer", MocapParameterizer("position")),
                ("downsampler", DownSampler(tgt_fps=30, keep_all=False)),
                # ("preprocess", CVAEPreprocess()),
                ("numpyfier", Numpyfier()),
            ]
        )
        piped_data = data_pipe.fit_transform([parsed_data])
        slicer = Slicer(window_size=240, overlap=0.5)
        piped_data = slicer.fit_transform(piped_data)
        return piped_data
    except:
        return None


def main():
    bvh_files = get_locomotion_files("../motionsynth_data/data/processed/cmu")
    print(f"Found {len(bvh_files)} files")

    all_windows = []

    max_workers = min(cpu_count(), len(bvh_files))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(
            tqdm(
                executor.map(process_bvh_file, bvh_files),
                total=len(bvh_files),
                desc="Processing",
            )
        )

    for result in results:
        if result is not None and result.shape[0] != 0:
            all_windows.append(result)

    final_data = np.concatenate(all_windows, axis=0)
    np.savez_compressed("./data/03_data.npz", clips=final_data)

    if not os.path.exists("./data/pose0.npy"):
        pose0 = np.load("./data/03_data.npz")["data"][0]
        pose0 = np.expand_dims(pose0, axis=0)
        np.save("./data/pose0.npy", pose0)


if __name__ == "__main__":
    main()
