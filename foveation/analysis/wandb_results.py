import wandb
import pandas as pd
from tqdm import tqdm


def flatten_dict(d, parent_key='', sep='.'):
    """
    Flatten a nested dictionary.

    Args:
        d (dict): The dictionary to flatten.
        parent_key (str): The base key string to use for nested keys.
        sep (str): The separator to use for concatenating keys.

    Returns:
        dict: A flat dictionary with dot-separated keys.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            continue
        else:
            items.append((new_key, v))
    return dict(items)


def collect_wandb(project_name="eliasmeinecke-uni-frankfurt/bachelorthesis", filters=None):
    api = wandb.Api()
    runs = api.runs(project_name, filters=filters)

    df = []
    for run in tqdm(runs):
        df.append({
            **{k: v for k, v in run.summary.items() if isinstance(v, (int, str, float))},
            **{'name': run.name, 'id': run.id},
            **flatten_dict(run.config)
        })
    return pd.DataFrame(df)


df = collect_wandb(filters={'group': 'crop'})