from pathlib import Path
import json
import shutil

if __name__ == '__main__':
    model_root = Path("/pfss/mlde/workspaces/mlde_wsp_PI_Roig/schaumloeffel/logs/models_elias")
    pre_train_root = model_root / "mocov3"
    linear_root = model_root / "linear"

    with open("trained_models_config.json", "r") as f:
        config = json.load(f)


    keep_pre_trained_models = [c['id'] for c in config]
    pre_trained_models = list(filter(lambda p: p.is_dir(), pre_train_root.iterdir()))
    pre_trained_models_remove = [p for p in pre_trained_models if p.name not in keep_pre_trained_models]

    if pre_trained_models_remove:
        input(f"Removing the following {len(pre_trained_models_remove)} pre-trained models: " + ", ".join([p.name for p in pre_trained_models_remove]) + "\nPress Enter to continue...")

        # for p in pre_trained_models_remove:
        #     shutil.rmtree(p)

    linear_config = []
    remove_linear_models = []
    remove_linear_models_config = []
    for linear_p in filter(lambda x: x.is_dir(), linear_root.iterdir()):
        with open(linear_p / "args.json", "r") as f:
            args = json.load(f)
            idx = Path(args['pretrained_feature_extractor']).parent.name

            dataset = args['data']['dataset']

            if idx in keep_pre_trained_models:
                linear_config.append(dict(pre_trained_id=idx, dataset=dataset, id=linear_p.name))
            else:
                remove_linear_models.append(linear_p)
                remove_linear_models_config.append(args)

    with open("linear_models_config.json", "w") as f:
        json.dump(linear_config, f, indent=4)

    if remove_linear_models:
        input(f"Removing the following {len(remove_linear_models)} linear models: " + ", ".join([p.name for p in remove_linear_models]) + "\nPress Enter to continue...")
        for args in remove_linear_models_config:
            print(args['name'])

        # for p in remove_linear_models:
        #     shutil.rmtree(p)



