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


def process_bvh_file(filepath):
    try:
        parser = BVHParser()
        parsed_data = parser.parse(filepath)

        data_pipe = Pipeline(
            [
                ("parameterizer", MocapParameterizer("position")),
                ("jointselector", JointSelector(JOINTS, include_root=False)),
                ("downsampler", DownSampler(tgt_fps=30, keep_all=False)),
                ("preprocess", AutoencoderPreprocess()),
                ("numpyfier", Numpyfier()),
            ]
        )
        piped_data = data_pipe.fit_transform([parsed_data])
        slicer = Slicer(window_size=160, overlap=0.5)
        piped_data = slicer.fit_transform(piped_data)
        return piped_data
    except:
        return None


def main():
    bvh_files = glob.glob(
        os.path.join("../motionsynth_data/data/processed/cmu", "*.bvh")
    )

    all_windows = []

    max_workers = min(cpu_count(), len(bvh_files))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(
            tqdm(
                executor.map(process_bvh_file, bvh_files),
                total=len(bvh_files),
                desc="Processing BVH files",
            )
        )

    for result in results:
        if result is not None and result.shape[0] != 0:
            all_windows.append(result)

    final_data = np.concatenate(all_windows, axis=0)
    np.savez_compressed("01_data.npz", clips=final_data)
    print(final_data.shape)


if __name__ == "__main__":
    main()
