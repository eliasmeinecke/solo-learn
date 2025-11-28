



def kwargs():
    
    # common parameters for all methods
    base_kwargs = {
        "backbone": "vit_tiny",
        "num_classes": 10,
        "cifar": True,

        # training
        "max_epochs": 1,
        "batch_size": 64,
        "num_workers": 4,

        # optimizer
        "optimizer": "adamw",
        "lr": 0.0005,
        "weight_decay": 0.00001,

        # scheduler
        "scheduler": "warmup_cosine",
        "warmup_epochs": 0,

        # data
        "data_dir": "data",
        "dataset": "cifar10",
        "train_dir": None,
        "val_dir": None,

        # mps gpu
        "devices": 1,
        "accelerator": "auto",

        # logging + name
        "name": "mocov3-cifar10-test",
    }

    # mocov3 specific parameters
    method_kwargs = {
        "proj_hidden_dim": 2048,
        "proj_output_dim": 256,
        "pred_hidden_dim": 4096,
        "temperature":0.2,
    }

    kwargs = {**base_kwargs, **method_kwargs}

    print(kwargs)

    print("Model creation successful!")
    

    


if __name__ == "__main__":
    kwargs()