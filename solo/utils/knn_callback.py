from typing import Dict, Any, Tuple

import lightning.pytorch as pl
import torch
import torch.distributed as dist
from lightning.pytorch.utilities.types import STEP_OUTPUT
from omegaconf import DictConfig
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from solo.data.classification_dataloader import prepare_datasets, prepare_separated_transforms
from solo.utils.knn import WeightedKNNClassifier

from foveation.factory import GazePredictor, setup_foveation, log_foveation_config


class KNNCallback(pl.Callback):
    def __init__(self, cfg: DictConfig, foveation_cfg, gpu_augmentation):
        self.cfg = cfg
        self.train_loader, self.test_loader = None, None
        self.T_train_gpu, self.T_val_gpu = None, None
        self.foveation_cfg=foveation_cfg
        self.gpu_augmentation=gpu_augmentation
        # Only build foveation if GPU augmentation is globally enabled
        self.foveation = setup_foveation(foveation_cfg) if gpu_augmentation else None        
        self.gaze_predictor = GazePredictor()
        self._foveation_debug_printed = False

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        
        log_foveation_config(self.foveation_cfg, context="knn", gpu_augmentation=self.gpu_augmentation)
        
        T_pre_train, T_train_gpu, T_pre_val, T_val_gpu = prepare_separated_transforms(self.cfg.dataset)
        
        self.T_train_gpu, self.T_val_gpu = T_train_gpu, T_val_gpu
        
        train_dataset, val_dataset = prepare_datasets(
            self.cfg.dataset,
            T_pre_train,
            T_pre_val,
            train_data_path=self.cfg.train_path,
            val_data_path=self.cfg.val_path,
            data_format=self.cfg.format
        )
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            drop_last=True,
            sampler=DistributedSampler(train_dataset, shuffle=False)
        )
        self.test_loader = DataLoader(
            val_dataset,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            drop_last=False,
            sampler=DistributedSampler(val_dataset, shuffle=False)
        )
        if isinstance(self.cfg.perform_every_n_batches, float):
            print("Estimated stepping batches", trainer.estimated_stepping_batches)
            self.cfg.perform_every_n_batches = int(trainer.estimated_stepping_batches * self.cfg.perform_every_n_batches / trainer.max_epochs)


    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self.cfg.perform_on_validation and trainer.current_epoch >= self.cfg.delay_epochs and not trainer.current_epoch%self.cfg.freq_epochs:
            self._run(trainer, pl_module)

    def on_test_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self.cfg.perform_on_test:
            self._run(trainer, pl_module)

    def on_train_batch_end(
            self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: STEP_OUTPUT, batch: Any,
            batch_idx: int
    ) -> None:
        if self.cfg.perform_every_n_batches is not None and batch_idx % self.cfg.perform_every_n_batches == 0 and batch_idx != 0:
            self._run(trainer, pl_module)

    def _run(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        if not trainer.sanity_checking and not trainer.fast_dev_run:
            torch.cuda.empty_cache()
            pl_module.eval()

            result = self.run(trainer, pl_module)

            torch.cuda.empty_cache()
            pl_module.train()

            for k, value in result.items():
                if hasattr(trainer.logger, 'log_metrics'):
                    trainer.logger.log_metrics({
                        f'knn/{self.cfg.dataset}_{k}_top1': value[0],
                        f'knn/{self.cfg.dataset}_{k}_top5': value[1]
                    }, step=trainer.global_step)
                else:
                    raise ValueError("Please use a logger that supports `log_metrics`")

    @torch.no_grad()
    def extract_features(self, loader: DataLoader, model: pl.LightningModule, mode: str = "train") -> Tuple[
        torch.Tensor, torch.Tensor]:
        bar = tqdm(loader, desc=f'{mode} KNN',
                   total=len(loader)) if self.cfg.verbose and model.local_rank == 0 else loader

        res_X, res_y = [], []
        for batch in bar:
            X, y = batch
            X = X.to(model.device, non_blocking=True)
            y = y.to(model.device, non_blocking=True)
            
            # GPU foveation
            if (self.foveation is not None):
                if (
                    not self._foveation_debug_printed
                    and (not dist.is_available()
                        or not dist.is_initialized()
                        or dist.get_rank() == 0)
                ):
                    print("\n[FOVEATION DEBUG] kNN foveation is ACTIVE on GPU\n")
                    self._foveation_debug_printed = True
            
                gaze, saliency = self.gaze_predictor(X)
                X = self.foveation(X, gaze, saliency)

            # GPU transform
            transform = (
                self.T_train_gpu if mode == "train"
                else self.T_val_gpu
            )

            if transform is not None:
                X = torch.stack([transform(X[i]) for i in range(X.shape[0])])
            
            outs = model(X)
            res_X.append(outs["feats"].detach())
            res_y.append(y.detach())
        res_X = torch.cat(res_X)
        res_y = torch.cat(res_y)
        return res_X, res_y

    def run(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> Dict:
        # extract train and test features
        X_train, y_train = self.extract_features(self.train_loader, pl_module, mode="train")
        X_test, y_test = self.extract_features(self.test_loader, pl_module, mode="test")

        # barrier to make sure all features are extracted
        trainer.strategy.barrier()

        result = {}
        for k in self.cfg.k:
            knn = WeightedKNNClassifier(k=k, T=self.cfg.T, distance_fx=self.cfg.distance_fx)
            knn(X_train, y_train, X_test, y_test)
            val_knn_acc1, val_knn_acc5 = knn.compute()
            result[k] = (val_knn_acc1, val_knn_acc5)
            del knn

        return result
