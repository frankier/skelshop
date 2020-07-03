from typing import Any, Dict, List

import numpy as np
import torch
import umap
from cycler import cycler
from matplotlib import pyplot as plt
from pytorch_lightning.core.lightning import LightningModule
from pytorch_metric_learning import losses, miners, testers
from sklearn.model_selection import train_test_split as single_train_test_split
from torch.utils.data import DataLoader, Subset

from . import pt_tb_monkey  # noqa
from .embed_skels import EMBED_SKELS
from .flex_st_gcn import FlexStGcn
from .graph import GraphAdapter
from .multilabel_softmax import NormalizedMultilabelSoftmaxLoss
from .pt_datasets import BodySkeletonDataset, DataPipeline, HandSkeletonDataset
from .utils import save_fig_np


class MetGcnLit(LightningModule):
    """
    Class for metric learning using mmskeleton GCNs using PyTorch Lightning.
    """

    # *Setup

    def __init__(
        self,
        data_path,
        mode="eval",
        graph="HAND",
        embed_size=64,
        loss="nsm",
        batch_size=32,
        lr=0.001,
        no_aug=False,
        vocab=None,
        include_score=False,
        body_labels=None,
    ):
        super().__init__()
        self.data_path = data_path
        self.skel_graph = EMBED_SKELS[graph]
        self.embed_size = embed_size
        self.loss_name = loss
        if graph == "HAND":
            self.dataset = HandSkeletonDataset(self.data_path, vocab=vocab)
        else:
            self.dataset = BodySkeletonDataset(
                self.data_path,
                vocab=vocab,
                powerset=loss == "psnsm",
                body_labels=body_labels,
            )
        self.mode = mode
        assert self.mode in ("eval", "prod")
        self.batch_size = batch_size
        self.lr = lr
        self.no_aug = no_aug
        self.include_score = include_score

        self.gcn = FlexStGcn(
            3 if self.include_score else 2,
            self.embed_size,
            GraphAdapter(self.skel_graph),
        )

        if self.mode == "prod":
            self.num_classes = self.dataset.CLASSES_TOTAL
        else:
            self.num_classes = self.dataset.CLASSES_TOTAL - len(
                HandSkeletonDataset.LEFT_OUT_EVAL
            )

        # Parameters are based on ones that seem reasonable given
        # https://github.com/KevinMusgrave/powerful-benchmarker
        # Could optimise...
        if self.loss_name in ("nsm", "psnsm"):
            self.loss = losses.NormalizedSoftmaxLoss(
                0.07, self.embed_size, self.num_classes
            )
        elif self.loss_name == "lbnsm":
            self.loss = NormalizedMultilabelSoftmaxLoss(
                0.07, self.embed_size, self.num_classes
            )
        else:
            assert self.loss_name == "msl"
            self.loss = losses.MultiSimilarityLoss(10, 50, 0.7)
            self.miner = miners.MultiSimilarityMiner(epsilon=0.5)

    def setup(self, stage):
        dataset = self.dataset
        if self.mode == "prod":
            self.train_dataset = dataset
            self.val_dataset = Subset(dataset, [])
            self.test_dataset = Subset(dataset, [])
            self.val_tester = None
        else:
            train_val_idxs = []
            train_val_clses = []
            test_idxs = []
            for idx in range(len(dataset)):
                cls = dataset[idx]["category_id"]
                if cls >= self.num_classes:
                    test_idxs.append(idx)
                else:
                    train_val_idxs.append(idx)
                    train_val_clses.append(cls)

            train_idxs, val_idxs = single_train_test_split(
                train_val_idxs, test_size=0.1, stratify=train_val_clses
            )

            # assign to use in dataloaders
            self.train_dataset = Subset(dataset, train_idxs)
            self.val_dataset = Subset(dataset, val_idxs)
            self.test_dataset = Subset(dataset, test_idxs)

            self.val_tester = self.setup_validation_tester()

        self.train_dataset = self.mk_data_pipeline(self.train_dataset)
        self.val_dataset = self.mk_data_pipeline(self.val_dataset)
        self.test_dataset = self.mk_data_pipeline(self.test_dataset)

    # *Common

    def batch_loss(self, batch):
        x, y = batch
        y_hat = self(x)
        if self.loss_name == "nsm":
            return self.loss(y_hat, y)
        else:
            assert self.loss_name == "msl"
            hard_pairs = self.miner(y_hat, y)
            return self.loss(y_hat, y, hard_pairs)

    def mk_data_pipeline(self, dataset):
        from mmskeleton.datasets.skeleton import (
            normalize_by_resolution,
            mask_by_visibility,
            simulate_camera_moving,
            transpose,
            to_tuple,
        )

        aug_steps: List[Dict[str, Any]] = []
        if not self.no_aug:
            aug_steps.extend((dict(stage=simulate_camera_moving),))
        if not self.include_score:
            aug_steps.append(
                dict(stage=lambda data: {**data, "data": data["data"][:2, :, :, :]})
            )
        return DataPipeline(
            dataset,
            [
                dict(stage=normalize_by_resolution),
                dict(stage=mask_by_visibility),
                *aug_steps,
                dict(stage=transpose, order=[0, 2, 1, 3]),
                dict(stage=to_tuple),
            ],
        )

    def mk_data_loader(self, dataset, shuffle=False):
        return DataLoader(
            dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=8,
        )

    # *Training

    def forward(self, x):
        return self.gcn(x)

    def training_step(self, batch, batch_idx):
        loss = self.batch_loss(batch)
        return {"loss": loss, "log": {"train_loss": loss}}

    def configure_optimizers(self):
        return torch.optim.SGD(
            self.parameters(),
            lr=self.lr,
            momentum=0.9,
            nesterov=True,
            weight_decay=0.0001,
        )
        # return torch.optim.Adam(self.parameters(), lr=self.lr)

    def train_dataloader(self):
        return self.mk_data_loader(self.train_dataset, shuffle=True)

    # *Validation

    def validation_step(self, batch, batch_idx):
        loss = self.batch_loss(batch)
        return {"val_loss": loss, "log": {"val_loss": loss}}

    def setup_validation_tester(self):
        return testers.GlobalEmbeddingSpaceTester(
            visualizer=umap.UMAP(),
            visualizer_hook=self.visualizer_hook,
            dataloader_num_workers=32,
        )

    def visualizer_hook(
        self, umapper, umap_embeddings, labels, split_name, keyname, *args
    ):
        # Plotting
        label_set = np.unique(labels)
        num_classes = len(label_set)
        plt.figure(figsize=(20, 15))
        plt.gca().set_prop_cycle(
            cycler(
                "color",
                [plt.cm.nipy_spectral(i) for i in np.linspace(0, 0.9, num_classes)],
            )
        )
        for i in range(num_classes):
            idx = labels == label_set[i]
            plt.plot(
                umap_embeddings[idx, 0], umap_embeddings[idx, 1], ".", markersize=1
            )
        # Log image
        self.logger.experiment.add_image(
            f"umap_plot_{split_name}_{keyname}",
            save_fig_np(plt.gcf()),
            self.global_step,
        )
        # Add embedding for projector / reprojection
        assert self.val_tester is not None
        embeddings, labels = self.val_tester.embeddings_and_labels[split_name]
        self.logger.experiment.add_embedding(
            embeddings,
            labels,
            global_step=self.global_step,
            tag="proj_{split_name}_{keyname}",
        )

    def validation_epoch_end(self, outputs):
        if self.val_tester is None:
            return
        val_loss_mean = torch.stack([x["val_loss"] for x in outputs]).mean()
        self.val_tester.test(
            {"train": self.train_dataset, "val": self.val_dataset,},
            self.current_epoch,
            self,
            splits_to_eval=["val"],
        )
        return {
            "val_loss": val_loss_mean,
            "global_embedding_space_accuracies": self.val_tester.all_accuracies,
        }

    def val_dataloader(self):
        return self.mk_data_loader(self.val_dataset)

    # *Testing

    def test_step(self, batch, batch_idx):
        # Everything is done by GlobalEmbeddingSpaceTester
        return {}

    def test_epoch_end(self, outputs):
        if self.mode == "prod":
            return
        tester = testers.GlobalEmbeddingSpaceTester(
            "compared_to_sets_combined",
            visualizer=umap.UMAP(),
            visualizer_hook=self.visualizer_hook,
            dataloader_num_workers=32,
        )
        tester.test(
            {
                "train": self.train_dataset,
                "val": self.val_dataset,
                "test": self.test_dataset,
            },
            self.current_epoch,
            self,
            ["test"],
        )
        return {"global_embedding_space_accuracies": tester.all_accuracies}

    def test_dataloader(self):
        return self.mk_data_loader(self.test_dataset)
