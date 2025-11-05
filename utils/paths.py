import os

from glob import glob


def get_image_and_mask_paths(data_path: str, image_folder: str = "image", mask_folder: str = "nodule_mask",
                             file_extension: str = ".nii.gz") -> tuple[list, list]:
    """
    Get all images and mask paths. The paths are sorted.

    Arguments:

    - data_path (str): The path to the dataset with the images and the masks.
    - image_folder (str): The name of the folder with the images (default: image).
    - mask_folder (str): The name of the folder with the masks (default: nodule_mask).
    - file_extension (str): The extension from the files (default: .nii.gz).

    Return

    A tuple with two list. The first list contains the image paths, the second list the mask paths.
    """
    img_paths = glob(os.path.join(data_path, image_folder, f"*{file_extension}"))

    if len(img_paths) == 0:
        raise ValueError(f"No images found in path {os.path.join(data_path, image_folder)}")

    img_paths = sorted(img_paths)
    mask_paths = []

    for img_path in img_paths:
        file_name = os.path.basename(img_path)
        mask_path = os.path.join(data_path, mask_folder, file_name)

        if not os.path.exists(mask_path) and not os.path.isfile(mask_path):
            print(f"No mask file found for image: {file_name}!")
            continue

        mask_paths.append(mask_path)

    assert len(img_paths) == len(mask_paths)

    return img_paths, mask_paths


def get_split_idx(paths: list, split_ratio: float = .8) -> int:
    """
    Returns the index to split the dataset. Dataset must be sorted and all the data from one patient is in the same
    split.

    Arguments:

    - paths (list[str]): The sorted paths to split
    - split_ratio (float): The ratio to split the paths (default: 0.8)

    Returns:

    The index to split the dataset
    """

    def get_id(file: str) -> str:
        return os.path.basename(file).split(".")[0].split("_")[0]

    ids = list(set(get_id(file=file) for file in paths))
    split_ids = int(split_ratio * len(ids))

    ids_train = ids[:split_ids]
    split_idx = 0
    for p in paths:
        if get_id(file=p) in ids_train:
            split_idx += 1

    return split_idx
