# Project RoboOrchard
#
# Copyright (c) 2024-2025 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

import argparse
import json
import logging
import os
from inspect import signature
from multiprocessing import set_start_method

import torch
from accelerate import Accelerator
from accelerate.state import AcceleratorState, is_initialized
from accelerate.utils import DataLoaderConfiguration, ProjectConfiguration
from utils import ActionMetric, load_checkpoint, load_config

from robo_orchard_lab.dataset.collates import collate_batch_dict
from robo_orchard_lab.dataset.dataset_wrapper import (
    DistributedBatchFlagSampler,
)
from robo_orchard_lab.pipeline import SimpleTrainer
from robo_orchard_lab.pipeline.batch_processor import SimpleBatchProcessor
from robo_orchard_lab.pipeline.hooks import (
    LossMovingAverageTrackerConfig,
    SaveCheckpointConfig,
    StatsMonitorConfig,
)
from robo_orchard_lab.utils import log_basic_config
from robo_orchard_lab.utils.torch import switch_model_mode

logger = logging.getLogger(__file__)


class ValidationLossTrainer(SimpleTrainer):
    """SimpleTrainer extension that always computes validation loss.

    It optionally computes action metrics when `self.metric` is provided,
    and can cap validation iteration count for faster debug cycles.
    """

    def __init__(self, *args, max_val_batches=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_val_batches = (
            max_val_batches if max_val_batches and max_val_batches > 0 else None
        )

    def _iter_named_losses(self, outputs, prefix=""):
        """Recursively collect scalar loss terms from nested model outputs."""
        if outputs is None:
            return
        if isinstance(outputs, dict):
            for name, value in outputs.items():
                full_name = f"{prefix}.{name}" if prefix else name
                if "loss" in name and value is not None and hasattr(
                    value, "mean"
                ):
                    yield full_name, value.mean().item()
                else:
                    yield from self._iter_named_losses(value, full_name)
            return
        if isinstance(outputs, (list, tuple)):
            for idx, value in enumerate(outputs):
                full_name = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
                yield from self._iter_named_losses(value, full_name)

    def _compute_validation_loss(self, batch):
        """Run one train-mode forward pass and return total/component losses."""
        with switch_model_mode(self.model, target_mode="train"):
            loss_outputs = self.model(batch)
        component_losses = {}
        for name, value in self._iter_named_losses(loss_outputs):
            component_losses[name] = component_losses.get(name, 0.0) + value
        if not component_losses:
            return None, {}
        return sum(component_losses.values()), component_losses

    @torch.no_grad()
    def eval(self):
        """Evaluate val set and log loss metrics (and optional action metrics)."""
        assert self.val_dataloader is not None, (
            "val_dataloader should not be None"
        )
        training = self.model.training
        self.model.eval()
        torch.cuda.empty_cache()
        if self.accelerator.is_main_process:
            logger.info("\n" + "=" * 50 + "BEGIN EVAL" + "=" * 50)
        total_loss = 0.0
        component_loss_totals = {}
        component_loss_counts = {}
        batches_with_loss = 0
        processed_batches = 0
        for val_step_id, batch in enumerate(self.val_dataloader):
            # Optional early stop for faster debug iterations.
            if (
                self.max_val_batches is not None
                and processed_batches >= self.max_val_batches
            ):
                break
            processed_batches += 1
            model_outputs = self.model(batch)
            if self.metric is not None:
                self.metric.update(batch, model_outputs)
            batch_loss, batch_component_losses = self._compute_validation_loss(
                batch
            )
            if batch_loss is not None:
                total_loss += batch_loss
                batches_with_loss += 1
                for name, value in batch_component_losses.items():
                    component_loss_totals[name] = (
                        component_loss_totals.get(name, 0.0) + value
                    )
                    component_loss_counts[name] = (
                        component_loss_counts.get(name, 0) + 1
                    )
            if (
                val_step_id + 1
            ) % 10 == 0 and self.accelerator.is_main_process:
                logger.info(f"eval: {val_step_id + 1}")
        self.accelerator.wait_for_everyone()
        metrics = {}
        if self.metric is not None:
            if "accelerator" in signature(self.metric.compute).parameters:
                metrics = self.metric.compute(accelerator=self.accelerator)
            else:
                metrics = self.metric.compute()
            if metrics is None:
                metrics = {}
        self.accelerator.wait_for_everyone()
        if batches_with_loss > 0:
            validation_loss = total_loss / batches_with_loss
            metrics["validation_loss"] = validation_loss
            component_items = []
            for name in sorted(component_loss_totals):
                component_avg = (
                    component_loss_totals[name] / component_loss_counts[name]
                )
                metrics[f"validation_{name}"] = component_avg
                component_items.append(f"{name}={component_avg:.6f}")
            if self.accelerator.is_main_process:
                logger.info(
                    f"total_validation_loss: {validation_loss:.6f} "
                    f"from {batches_with_loss} batches"
                )
                if component_items:
                    logger.info(
                        "validation_loss_components: "
                        + ", ".join(component_items)
                    )
        if (
            self.max_val_batches is not None
            and self.accelerator.is_main_process
            and processed_batches >= self.max_val_batches
        ):
            logger.info(
                f"validation limited to {self.max_val_batches} batches"
            )
        if metrics:
            eval_step = self.trainer_progress_state.global_step_id
            # Prefix with `val/` so curves are separated from training metrics.
            self.accelerator.log(
                {f"val/{k}": v for k, v in metrics.items()},
                step=eval_step,
            )
            if self.accelerator.is_main_process:
                logger.info(f"tensorboard_eval_step: {eval_step}")
        if self.metric is not None:
            self.metric.reset()
        torch.cuda.empty_cache()
        self.model.train(training)
        return metrics


class MyBatchProcessor(SimpleBatchProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, model, batch):
        output = model(batch)
        loss = sum([y.mean() for x, y in output.items() if "loss" in x])
        return output, loss


def main(args, accelerator):
    if_cluster = os.environ.get("CLUSTER") is not None
    if accelerator.is_main_process:
        import shutil

        shutil.copytree(
            "configs",
            os.path.join(args.workspace, "configs"),
            dirs_exist_ok=True,
        )

    config = load_config(args.config)
    build_model = config.build_model
    build_dataset = config.build_training_dataset
    build_validation_dataset = config.build_validation_dataset
    build_optimizer = config.build_optimizer
    build_processors = config.build_processors
    config = config.config

    # export data processors
    if accelerator.is_main_process:
        processors = build_processors(config)
        for dataset_name, processor in processors.items():
            processor.save(args.workspace, f"{dataset_name}_processor.json")

    if args.kwargs is not None:
        if os.path.isfile(args.kwargs):
            kwargs = json.load(open(args.kwargs, "r"))
        else:
            kwargs = json.loads(args.kwargs)
        config.update(kwargs)

    if accelerator.is_main_process:
        logger.info("\n" + json.dumps(config, indent=4))

    model = build_model(config)

    num_workers = config.get("num_workers", 4)
    if not args.eval_only:
        train_dataset = build_dataset(config)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            num_workers=num_workers,
            pin_memory=False,
            collate_fn=collate_batch_dict,
            persistent_workers=num_workers > 0,
            batch_sampler=DistributedBatchFlagSampler(
                train_dataset,
                config["batch_size"],
                drop_last=True,
                dataset_sample_weights=config.get("dataset_sample_weights"),
            ),
            # in_order=False,
        )
        optimizer, lr_scheduler = build_optimizer(config, model)
    else:
        train_dataloader = optimizer = lr_scheduler = None

    trainable_param = 0
    non_trainable_param = 0
    for param in model.parameters():
        if param.requires_grad:
            trainable_param += param.numel()
        else:
            non_trainable_param += param.numel()
    total_param = trainable_param + non_trainable_param
    logger.info(
        f"number of parameters: {total_param / 10**6:.2f}M, "
        f"trainable: {trainable_param / 10**6:.2f}M, "
        f"non-trainable: {non_trainable_param / 10**6:.2f}M"
    )

    accelerator.register_save_state_pre_hook(
        model.accelerator_save_state_pre_hook
    )
    load_checkpoint(model, config.get("checkpoint"), accelerator)

    val_dataset = build_validation_dataset(config)
    if val_dataset is not None:
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=False,
            batch_size=config["batch_size"],
            collate_fn=collate_batch_dict,
            persistent_workers=num_workers > 0,
        )
        pred_steps = config.get("pred_steps", 64)
        metric = (
            ActionMetric(
                eval_horizons=[pred_steps // 4, pred_steps // 2, pred_steps],
            )
            if args.do_eval_actions
            else None
        )
    else:
        val_dataloader = None
        metric = None

    trainer = ValidationLossTrainer(
        model=model,
        dataloader=train_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
        grad_clip_mode="norm",
        grad_max_norm=10,
        batch_processor=MyBatchProcessor(need_backward=True),
        hooks=[
            StatsMonitorConfig(
                step_log_freq=config["step_log_freq"],
            ),
            LossMovingAverageTrackerConfig(
                step_log_freq=config["step_log_freq"]
            ),
            SaveCheckpointConfig(
                save_step_freq=config.get("save_step_freq"),
                save_epoch_freq=config.get("save_epoch_freq"),
            ),
        ],
        max_step=config.get("max_step"),
        step_eval_freq=config.get("save_step_freq"),
        #step_eval_freq=config.get("step_log_freq"),
        lr_scheduler_step_at="step",
        resume_from=config.get("resume_from"),
        resume_share_dir=(
            "/job_data/resume_from" if if_cluster else "./resume_from"
        ),
        val_dataloader=val_dataloader,
        metric=metric,
        max_val_batches=args.max_val_batches,
    )
    if args.eval_only:
        assert val_dataset is not None, (
            "The validation dataset must be specified when eval_only=True."
        )
        trainer.eval()
    else:
        trainer()
    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--workspace", type=str, default="./workspace")
    parser.add_argument("--logging_dir", type=str, default=None)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--kwargs", type=str, default=None)
    parser.add_argument(
        "--do-eval-actions",
        dest="do_eval_actions",
        action="store_true",
        help="enable action metrics during validation",
    )
    parser.add_argument(
        "--no-eval-actions",
        dest="do_eval_actions",
        action="store_false",
        help="skip action metrics and only report validation loss",
    )
    parser.add_argument(
        "--max-val-batches",
        type=int,
        default=None,
        help="maximum number of validation batches per eval; unset or <=0 means full validation",
    )
    parser.set_defaults(do_eval_actions=False)
    args = parser.parse_args()

    if args.logging_dir is None:
        args.logging_dir = os.path.join(args.workspace, "logs")

    os.makedirs(args.workspace, exist_ok=True)
    os.makedirs(args.logging_dir, exist_ok=True)
    accelerator = Accelerator(
        log_with="tensorboard",
        step_scheduler_with_optimizer=False,
        project_config=ProjectConfiguration(
            project_dir=args.workspace,
            logging_dir=args.logging_dir,
            automatic_checkpoint_naming=True,
            total_limit=3,
        ),
        dataloader_config=DataLoaderConfiguration(
            use_seedable_sampler=True,
        ),
    )
    accelerator.init_trackers("tensorboard")

    log_basic_config(
        format="%rank %(asctime)s %(levelname)s %(filename)s:%(lineno)d | %(message)s",  # noqa: E501
        level=logging.INFO,
    )
    logger.info(f"Save config to workspace dir {args.workspace}")
    logger.info(f"TensorBoard logging dir: {args.logging_dir}")
    logger.info(f"if accelerator initialized:{is_initialized()}")
    logger.info(f"accelerator state: {AcceleratorState._shared_state}")
    set_start_method("spawn", force=True)
    main(args, accelerator)
