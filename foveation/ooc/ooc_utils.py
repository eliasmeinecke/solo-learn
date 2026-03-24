from pathlib import Path
import torch
import json
import pandas as pd
import torchvision.models as models
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as v2

from solo.methods.base import BaseMethod
from foveation.ooc.ooc_data import OOCOriginalDataset, OOCInpaintedDataset, OOCObjectOnlyDataset, OOCShuffledDataset



with open(Path("trained_models_config.json")) as f:
    MODEL_CONFIGS = json.load(f)

with open(Path("linear_models_config.json")) as f:
    LINEAR_CONFIGS = json.load(f)
    
with open(Path("best_classifiers.json")) as f:
    BEST_CLASSIFIERS = json.load(f)
    
    
class IdentityFoveation(torch.nn.Module):
    def forward(self, img, gaze):
        return img


def load_data(dataset_name):

    root = Path("/home/data/elias/ImageNet-OOC1k_flattened")

    T_pre = v2.Compose([
        v2.Resize(540),
        v2.ToImage(),
        v2.ToDtype(torch.uint8)
    ])

    common_kwargs = dict(
        root=root,
        transform=T_pre
    )

    dataset_map = {
        "original": OOCOriginalDataset,
        "inpainted": OOCInpaintedDataset,
        "object": OOCObjectOnlyDataset,
        "ooc": OOCShuffledDataset,
    }

    if dataset_name not in dataset_map:
        raise ValueError(dataset_name)

    dataset = dataset_map[dataset_name](**common_kwargs)

    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    

def find_model_config(model_name):
    for m in MODEL_CONFIGS:
        if m["name"].endswith(model_name):
            return m
    raise ValueError(f"Model not found: {model_name}")


def get_ckpt_path(model_cfg):
    run_id = model_cfg["id"]
    name = model_cfg["name"]

    base = Path("/home/data/elias/archive_extracted/mocov3")

    ckpt = base / run_id / f"{name}-{run_id}-ep=last.ckpt"

    if not ckpt.exists():
        raise FileNotFoundError(ckpt)

    return ckpt


def load_mocov3_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["state_dict"]

    # EXACT SAME LOGIC as main_linear
    for k in list(state.keys()):
        if "encoder" in k:
            state[k.replace("encoder", "backbone")] = state[k]

        if "backbone" in k:
            state[k.replace("backbone.", "")] = state[k]

        del state[k]

    backbone_model = BaseMethod._BACKBONES["resnet50"]

    backbone = backbone_model(method="mocov3")
    
    backbone.fc = torch.nn.Identity()
    
    backbone.load_state_dict(state, strict=False)

    return backbone


def find_linear_head(pretrained_id):
    matches = [
        x for x in LINEAR_CONFIGS
        if x["pre_trained_id"] == pretrained_id and x["dataset"] == "imagenet_42"
    ]

    if len(matches) == 0:
        raise ValueError(f"No linear head for {pretrained_id} (imagenet_42)")

    if len(matches) > 1:
        print("[WARNING] Multiple heads found, taking first")

    return matches[0]


def get_linear_ckpt_path(linear_cfg, model_cfg):
    linear_id = linear_cfg["id"]
    model_name = model_cfg["name"]

    base = Path("/home/data/elias/linear_extracted/linear")

    ckpt = base / linear_id / f"debug_imagenet_42_{model_name}-{linear_id}-ep=last.ckpt"
    # have to check for debug and no debug prefix for later!!

    if not ckpt.exists():
        raise FileNotFoundError(ckpt)

    return ckpt


def find_key(state_dict, target):
    for k in state_dict.keys():
        if k.endswith(target):
            return k
    raise KeyError(target)


def load_linear_head(ckpt_path, pretrained_id):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["state_dict"]

    best_key = BEST_CLASSIFIERS[pretrained_id]

    weight_suffix = f"{best_key}.linear.weight"
    bias_suffix   = f"{best_key}.linear.bias"

    weight_key = find_key(state_dict, weight_suffix)
    bias_key   = find_key(state_dict, bias_suffix)

    linear = torch.nn.Linear(2048, 1000)
    linear.weight.data = state_dict[weight_key]
    linear.bias.data   = state_dict[bias_key]

    return linear


class FullModel(torch.nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        feats = self.backbone(x)

        # resnet might give [B,2048,1,1]
        if feats.ndim == 4:
            feats = feats.flatten(1)

        return self.head(feats)
    

def load_model(model_name):

    # dummy-model
    if model_name == "dummy":
        return models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    
    # find backbone config
    model_cfg = None
    for m in MODEL_CONFIGS:
        if model_name in m["name"]:
            model_cfg = m
            break

    if model_cfg is None:
        raise ValueError(f"Model not found: {model_name}")

    # backbone
    ckpt_path = get_ckpt_path(model_cfg)
    backbone = load_mocov3_model(ckpt_path)
    backbone.eval()

    # linear head
    linear_cfg = find_linear_head(model_cfg["id"])
    linear_ckpt = get_linear_ckpt_path(linear_cfg, model_cfg)
    head = load_linear_head(linear_ckpt, model_cfg["id"])

    print(f"[Model] Backbone: {model_cfg['name']}")
    print(f"[Model] Linear head trained on: {linear_cfg['dataset']}")

    model = FullModel(backbone, head)
    model.eval()

    return model


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