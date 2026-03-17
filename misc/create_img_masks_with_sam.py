import json
import os

import torch
from sam3.eval.postprocessors import PostProcessImage
from sam3.model.box_ops import box_xywh_to_cxcywh
from sam3.model.position_encoding import PositionEmbeddingSine
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model.utils.misc import copy_data_to_device
from sam3.model_builder import build_sam3_image_model
from sam3.train.data.collator import collate_fn_api as collate
from sam3.train.data.sam3_image_dataset import InferenceMetadata, FindQueryLoaded, Image as SAMImage, Datapoint
from sam3.train.transforms.basic_for_api import ComposeAPI, RandomResizeAPI, ToTensorAPI, NormalizeAPI
from sam3.visualization_utils import draw_box_on_image, normalize_bbox, plot_results
from torch.utils.data import DataLoader
from tqdm import tqdm

from solo.data.classification_dataloader import prepare_datasets

os.environ["HF_HUB_CACHE"] = "/pfss/mlde/workspaces/mlde_wsp_PI_Roig/shared/llm"
os.environ["HF_TOKEN"] = "" # <- add token here

transform = ComposeAPI(
    transforms=[
        RandomResizeAPI(sizes=1008, max_size=1008, square=True, consistent_transform=False),
        ToTensorAPI(),
        NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)


def create_datapoint(idx, pil_image, text_query):
    """ A datapoint is a single image on which we can apply several queries at once. """

    w, h = pil_image.size
    dp = Datapoint(find_queries=[FindQueryLoaded(
        query_text=text_query,
        image_id=0,
        object_ids_output=[],  # unused for inference
        is_exhaustive=True,  # unused for inference
        query_processing_order=0,
        inference_metadata=InferenceMetadata(
            coco_image_id=idx,
            original_image_id=idx,
            original_category_id=1,
            original_size=[h, w],
            object_id=0,
            frame_index=0,
        )
    )], images=[SAMImage(data=pil_image, objects=[], size=[h, w])])

    return transform(dp)


class Wrapper:
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        img, label = self.dataset[item]
        return create_datapoint(item, img, self.dataset.target_2_class_name[label].replace("_", " "))


def run(model, postprocessor, dataloader, out_path):
    save_every = 100
    result = {}
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            batch = copy_data_to_device(batch, torch.device("cuda"), non_blocking=True)
            output = model(batch)
            processed_results = postprocessor.process_results(output, batch.find_metadatas)

            for idx, data in processed_results.items():
                filename = dataloader.dataset.dataset.mapper.iloc[idx].filename

                result[idx] = {'rle': data['masks_rle'][0], 'bbox': data['boxes'][0].float().cpu().numpy().tolist(),
                               'filename': filename}

            if (i + 1) % save_every == 0:
                with open(out_path, "w") as f:
                    json.dump(result, f)

    with open(out_path, "w") as f:
        json.dump(result, f)


def setup():
    model = build_sam3_image_model()

    postprocessor = PostProcessImage(
        max_dets_per_img=1,
        # if this number is positive, the processor will return topk. For this demo we instead limit by confidence, see below
        iou_type="segm",  # we want masks
        use_original_sizes_box=True,  # our boxes should be resized to the image size
        use_original_sizes_mask=True,  # our masks should be resized to the image size
        convert_mask_to_rle=True,
        # the postprocessor supports efficient conversion to RLE format. In this demo we prefer the binary format for easy plotting
        detection_threshold=-1,  # Only return confident detections
        to_cpu=False,
    )

    train_ds, val_ds = prepare_datasets(
        "imagenet_42",
        train_data_path="/pfss/mlde/workspaces/mlde_wsp_PI_Roig/shared/datasets/",
        val_data_path="/pfss/mlde/workspaces/mlde_wsp_PI_Roig/shared/datasets/",
        T_train=lambda x: x,
        T_val=lambda x: x,
    )
    wrapped_train_ds, wrapped_val_ds = Wrapper(train_ds), Wrapper(val_ds)

    train_dl = DataLoader(wrapped_train_ds, batch_size=16, num_workers=8, shuffle=False,
                          collate_fn=lambda x: collate(x, dict_key="dummy")["dummy"])
    val_dl = DataLoader(wrapped_val_ds, batch_size=16, shuffle=False, num_workers=8,
                        collate_fn=lambda x: collate(x, dict_key="dummy")["dummy"])

    return model, postprocessor, train_dl, val_dl


if __name__ == '__main__':
    model, postprocessor, train_dl, val_dl = setup()
    # run(model, postprocessor, val_dl, "imagenet_val_masks.json")
    run(model, postprocessor, train_dl, "imagenet_train_masks.json")
