from pathlib import Path
import json
import shutil

if __name__ == '__main__':
    root = Path("/pfss/mlde/workspaces/mlde_wsp_PI_Roig/schaumloeffel/logs/ego4d/mocov3")



    for model_p in filter(lambda x: x.is_dir(), root.iterdir()):
        ckpts = [p for p in model_p.iterdir() if p.is_file() and p.name.endswith(".ckpt")]

        if not ckpts:
            continue
        if len(ckpts) == 1:
            continue

        found_last = False
        for ckpt in ckpts:
            if "last" in ckpt.name:
                found_last = True
                break

        if found_last:
            for ckpt in ckpts:
                if "last" not in ckpt.name:
                    print(f"Removing {ckpt} since last.ckpt exists")
                    ckpt.unlink()

