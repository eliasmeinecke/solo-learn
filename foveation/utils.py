from pathlib import Path
import json
from PIL import Image
from pycocotools import mask as mask_utils
import torch
import torchvision.models as models
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision.transforms import InterpolationMode
import torchvision.transforms.v2 as v2

from solo.methods.base import BaseMethod

from foveation.factory import setup_exact_foveation

IMAGENET_VAL_PATH = "/home/data/elias/ImageNet/val"
GAZE_JSON_PATH = "/home/data/elias/imagenet_sam_masks/imagenet_val_gaze_only.json"
MASK_JSON_PATH = "/home/data/elias/imagenet_sam_masks/imagenet_val_masks_with_center.json"

with open(Path("trained_models_config.json")) as f:
    MODEL_CONFIGS = json.load(f)

with open(Path("linear_models_config.json")) as f:
    LINEAR_CONFIGS = json.load(f)
    
T_POST = v2.Compose([
        v2.Resize(256),
        #v2.Resize(256, interpolation=InterpolationMode.BICUBIC, antialias=True),
        v2.CenterCrop(224),
        v2.ToImage(),
        v2.ToDtype(
            torch.float32,
            scale=True
        ),
        v2.Normalize(
            mean=[0.485,0.456,0.406],
            std=[0.229,0.224,0.225]
        )
    ])

class IdentityFoveation(torch.nn.Module):
    def forward(self, img, gaze):
        return img
    

def get_gaze_by_filename_map():
    with open(GAZE_JSON_PATH, "r") as f:
            gaze_data = json.load(f)
    gaze_by_filename = {v["filename"]: v for v in gaze_data.values()}
    return gaze_by_filename


def load_imagenet_class_map():
    with open("imagenet_class_index.json") as f:
        data = json.load(f)
    # Format:
    # {"0": ["n01440764", "tench"], ...}
    mapping = {}
    for idx, (synset, _) in data.items():
        mapping[synset] = int(idx)
    return mapping


# SETUP MODEL + FOVEATION
def build_model_and_foveation(model_name, device):
    model = load_model(model_name).to(device)
    model.eval()
    if model_name in ["base", "dummy"]:
        foveation = IdentityFoveation()
    else:
        foveation = setup_exact_foveation(model_name)
    foveation = foveation.to(device)
    return model, foveation


class ImageNetGazeDataset(Dataset):
    def __init__(self, root, transform=None):
        self.dataset = ImageFolder(root=root)
        self.transform = transform
        self.gaze_map = get_gaze_by_filename_map()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        path, label = self.dataset.samples[idx]
        filename = Path(path).name
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        # --- gaze + area ---
        dp = self.gaze_map.get(filename, None)
        if dp is not None:
            gaze_rel = torch.tensor([
                    dp["centroid"]["x_rel"],
                    dp["centroid"]["y_rel"]
                ], dtype=torch.float32)
            area = torch.tensor(dp["area"], dtype=torch.float32)
        else:
            gaze_rel = torch.tensor([0.5, 0.5], dtype=torch.float32)
            area = torch.tensor(0.0, dtype=torch.float32)
        return img, label, gaze_rel, area


class ImageNetMaskLoader:
    def __init__(self):
    
        with open(MASK_JSON_PATH, "r") as f:
            data = json.load(f)

        self.masks_by_filename = {
            v["filename"]: v for v in data.values()
        }

    def get_mask(self, filename):
        entry = self.masks_by_filename.get(filename, None)
        if entry is None:
            return None
        # --- decode RLE ---
        mask = mask_utils.decode(entry["rle"])  # (H, W)
        return mask.astype(bool)

    def get_area(self, filename):
        entry = self.masks_by_filename.get(filename, None)
        if entry is None:
            return 0.0
        return entry.get("area_mask_rel", 0.0)

    def get_centroid(self, filename):
        entry = self.masks_by_filename.get(filename, None)
        if entry is None:
            return None
        return (
            entry["centroid"]["x_rel"],
            entry["centroid"]["y_rel"]
        )
    
    
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
        if x["pre_trained_id"] == pretrained_id
    ]
    if len(matches) == 0:
        raise ValueError(f"No linear head for {pretrained_id}")
    # gaze_imagenet > imagenet_42 (could be left out now but just to make sure)
    preferred_order = ["gaze_imagenet", "imagenet_42"]
    for dataset_name in preferred_order:
        for m in matches:
            if m["dataset"] == dataset_name:
                return m
    # fallback if new linear heads are added
    print("[WARNING] No preferred dataset found, taking first available")
    return matches[0]


def get_linear_ckpt_path(linear_cfg):
    linear_id = linear_cfg["id"]
    base = Path("/home/data/elias/linear_extracted/linear")
    run_dir = base / linear_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Run dir not found: {run_dir}")
    ckpts = list(run_dir.glob("*-ep=last.ckpt"))
    if len(ckpts) == 0:
        raise FileNotFoundError(f"No checkpoint found in {run_dir}")
    if len(ckpts) > 1:
        print(f"[WARNING] Multiple checkpoints found in {run_dir}, taking first")
    ckpt_path = ckpts[0]
    # get clean name for debug printing
    # remove .ckpt
    name = ckpt_path.stem  
    # remove "-ep=last"
    name = name.replace("-ep=last", "")
    # remove "-<id>"
    if name.endswith(f"-{linear_id}"):
        name = name[: -(len(linear_id) + 1)]
    return ckpt_path, name


def find_key(state_dict, target):
    for k in state_dict.keys():
        if k.endswith(target):
            return k
    raise KeyError(target)


def load_linear_head(ckpt_path, pretrained_id):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["state_dict"]
    # all classifiers peak for same learning rate
    weight_suffix = "classifier-lr_2:00000000.linear.weight"
    bias_suffix = "classifier-lr_2:00000000.linear.bias"
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
    linear_ckpt, linear_name = get_linear_ckpt_path(linear_cfg)
    head = load_linear_head(linear_ckpt, model_cfg["id"])
    print(f"[Model] Backbone: {model_cfg['name']}")
    print(f"[Model] Linear head: {linear_name}")
    model = FullModel(backbone, head)
    model.eval()
    return model