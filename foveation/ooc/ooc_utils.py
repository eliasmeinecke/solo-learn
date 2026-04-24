from pathlib import Path
import pandas as pd
from torch.utils.data import DataLoader
from torchvision.transforms import PILToTensor

from foveation.ooc.ooc_data import OOCOriginalDataset, OOCInpaintedDataset, OOCObjectOnlyDataset, OOCShuffledDataset

OOC_DATASETS = ["original", "inpainted", "object", "ooc"]


def load_data(dataset_name, variant=None):

    if dataset_name not in OOC_DATASETS:
        raise ValueError(dataset_name)
    
    root = Path("/home/data/elias/ImageNet-OOC1k_flattened")

    T_pre = PILToTensor()

    common_kwargs = dict(
        root=root,
        transform=T_pre
    )
        
    if dataset_name == "original":
        dataset = OOCOriginalDataset(**common_kwargs)    
    elif dataset_name == "inpainted":
        
        if variant == "random_gaze":
            gaze_mode = "random"
        elif variant == "central_gaze":
            gaze_mode = "central"
        else:
            gaze_mode = "mask"
            
        dataset = OOCInpaintedDataset(
            **common_kwargs,
            gaze_mode=gaze_mode
        )
    elif dataset_name == "object":
        
        if variant == "black":
            background = "black"
        elif variant == "white":
            background = "white"
        elif variant == "gray":
            background = "gray"
        else:
            background = "imagenet"
            
        dataset = OOCObjectOnlyDataset(
            **common_kwargs,
            background=background
        )
    elif dataset_name == "ooc":
        dataset = OOCShuffledDataset(**common_kwargs)
    else:
        raise ValueError(dataset_name)
    
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)


def export_ooc_image_ids():
    """
    Reads ImageNet-OOC metadata.csv and exports image filenames
    to a txt file for filtering ImageNet val.
    """
    
    metadata_csv = "ImageNet-OOC1k_release/metadata.csv"
    output_txt = "ooc_image_ids.txt"

    df = pd.read_csv(metadata_csv)

    image_ids = df["image_id"].tolist()

    # convert to ImageNet filenames
    filenames = [f"{img_id}.JPEG" for img_id in image_ids]

    output_txt = Path(output_txt)

    with open(output_txt, "w") as f:
        for name in filenames:
            f.write(name + "\n")

    print(f"Saved {len(filenames)} image ids to {output_txt}")
    
    
if __name__ == "__main__":
    print("-------------------------------")
    # export_ooc_image_ids()
    print("-------------------------------")