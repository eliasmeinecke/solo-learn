import argparse
import subprocess
from pathlib import Path

import pandas as pd
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default="trained_models_config.json")
    parser.add_argument('--env', type=str, default="42")
    parser.add_argument('--devices', type=int, default=4)
    parser.add_argument('--epoch', type=str, default="last")
    args = parser.parse_args()

    meta = pd.read_json(args.ckpt)

    if args.env == "42":
        root = Path("/pfss/mlde/workspaces/mlde_wsp_PI_Roig/schaumloeffel/logs/models_elias/mocov3/")
    else:
        raise NotImplemented(f"env {args.env} not implemented")

    datasets = [
        'imagenet_42',
        'imagenet100_42',
        # 'cifar100_224',
        # 'cifar10_224',
        'COIL100',
        'DTD',
        # 'Flowers102',
        # 'FGVCAircraft',
        # 'OxfordIIITPet',
        # 'StanfordCars',
        # 'STL10',
        'toybox',
        'core50',
        'imagenet1pct_42',
        'imagenet10pct_42',
    ]

    job_type = "linear_probe_grid"
    config = "mocov3_resnet_linear_grid.yaml"

    base_cmd = f"python main_linear.py --config-path scripts/linear/ --config-name {config}"

    cmd_args = ("++name=\"{name}\" ++pretrained_feature_extractor=\"{ft}\" ++data.dataset={ds} ++wandb.job_type={job_type} "
            "++devices={devices} ++optimizer.batch_size={bs} ++checkpoint.enabled={store_ckpt}")

    bar = tqdm(total=len(datasets) * len(meta))
    for dataset in datasets:
        if dataset == "COIL100":
            devices = 2
            bs = 16
        else:
            devices = args.devices
            bs = 128

        store_ckpt = True if dataset in ['imagenet_42', 'Places365_h5', "imagenet1pct_42", "imagenet10pct_42"] else False

        for i, row in meta.iterrows():
            name = (dataset + '_' + row['model_name']).replace("=", "\=")

            ckpt_name = f"{row['id']}/{row['model_name']}-{row['id']}-ep={args.epoch}.ckpt"
            ckpt_path = root / ckpt_name
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Checkpoint {ckpt_path} does not exist.")
            ckpt_path = str(ckpt_path).replace("=", "\=")

            cmd = base_cmd + " " + cmd_args.format(name=name, ft=ckpt_path, ds=dataset, job_type=job_type,
                                                   devices=devices, bs=bs, store_ckpt=store_ckpt)

            print(f"Running command: {cmd}")
            subprocess.run(cmd, shell=True)
            bar.update(1)
            break

