# This code is adapted from # Adapted from https://github.com/zhengyiluo/phc/tree/h1_phc

import os
from pathlib import Path

import numpy as np
import typer

from data.scripts.retargeting.fit_smpl_shape import create_smpl_shape
from data.scripts.retargeting.fit_smpl_motion import convert_motions


def main(
    amass_root_dir: Path,
    force_remake: bool = False,
    humanoid_type: str = "h1",
):
    folder_names = [
        f.path.split("/")[-1] for f in os.scandir(amass_root_dir) if f.is_dir()
    ]

    shape_file = Path(f"data/{humanoid_type}/shape_optimized_v1.pt")

    if not shape_file.exists():
        create_smpl_shape(humanoid_type)

        if not shape_file.exists():
            raise FileNotFoundError(
                f"Failed to generate optimized shape file for {humanoid_type}"
            )
        else:
            print(f"Successfully generated optimized shape file for {humanoid_type}")

    file_list = []
    all_files = []
    for folder_name in folder_names:
        if "h1" in folder_name or "retarget" in folder_name or "smpl" in folder_name:
            # Ignore folders where we store converted motions
            continue

        data_dir = amass_root_dir / folder_name
        output_dir = amass_root_dir / f"{folder_name}-{humanoid_type}"

        print(f"Processing subset {folder_name}")
        os.makedirs(output_dir, exist_ok=True)

        files = [
            f
            for f in Path(data_dir).glob("**/*.npz")
            if (f.name != "shape.npz" and "stagei.npz" not in f.name)
        ]
        print(f"Processing {len(files)} files")

        files.sort()

        for filename in files:
            relative_path_dir = filename.relative_to(data_dir).parent
            relative_path_dir.mkdir(exist_ok=True, parents=True)

            outpath = (
                output_dir
                / relative_path_dir
                / filename.name.replace(".npz", ".npy")
                .replace("-", "_")
                .replace(" ", "_")
                .replace("(", "_")
                .replace(")", "_")
            )
            all_files.append(outpath)
            # Check if the output file already exists
            if not force_remake and outpath.exists():
                print(f"Skipping {filename} as it already exists.")
                continue

            file_list.append((filename, outpath))

            # Create the output directory if it doesn't exist
            os.makedirs(output_dir / relative_path_dir, exist_ok=True)

    from multiprocessing import Pool, set_start_method

    set_start_method("spawn")
    num_jobs = 2
    chunk = np.ceil(len(file_list) / num_jobs).astype(int)

    print(
        f"Processing {len(file_list)} out of {len(all_files)} files with {num_jobs} jobs"
    )

    file_list = [file_list[i : i + chunk] for i in range(0, len(file_list), chunk)]
    job_args = [(humanoid_type, file_list[i]) for i in range(len(file_list))]
    if num_jobs == 1:
        convert_motions(humanoid_type, file_list[0])
    else:
        try:
            pool = Pool(num_jobs)  # multi-processing
            pool.starmap(convert_motions, job_args)
            pool.close()
            pool.join()
        except KeyboardInterrupt:
            pool.terminate()
            pool.join()


if __name__ == "__main__":
    typer.run(main)
