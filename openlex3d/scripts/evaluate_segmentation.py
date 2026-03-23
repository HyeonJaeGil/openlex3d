#!/usr/bin/env python
# -*- coding: UTF8 -*-
# PYTHON_ARGCOMPLETE_OK

import logging
from time import perf_counter

import hydra
from omegaconf import DictConfig
from pathlib import Path
from loguru import logger as loguru_logger

import openlex3d.core.metric as metric  # noqa
from openlex3d import get_path
from openlex3d.core.evaluation import (
    compute_feature_to_prompt_similarity,
    get_label_from_logits,
)
from openlex3d.core.io import load_predicted_features, load_prompt_list, save_results
from openlex3d.datasets import load_dataset
from openlex3d.models import load_model

logger = logging.getLogger(__name__)


class _LoguruHandler(logging.Handler):
    def emit(self, record):
        try:
            level = loguru_logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        loguru_logger.opt(exception=record.exc_info).log(level, record.getMessage())


def _configure_loguru_logging():
    root_logger = logging.getLogger()
    root_logger.handlers = [_LoguruHandler()]
    root_logger.setLevel(logging.INFO)


@hydra.main(
    version_base=None, config_path=f"{get_path()}/config", config_name="eval_segmentation"
)
def main(config: DictConfig):
    _configure_loguru_logging()
    overall_start = perf_counter()
    # Load dataset
    t0 = perf_counter()
    gt_cloud, gt_ids, openlex3d_gt_handler = load_dataset(
        config.dataset, load_openlex3d=True
    )
    logger.info(f"Loaded dataset in {perf_counter() - t0:.3f}s")

    # Run evaluation
    if config.evaluation.type == "features":
        # Load language model
        t0 = perf_counter()
        model = load_model(config.model)
        logger.info(f"Loaded language model in {perf_counter() - t0:.3f}s")

        # Load predicted features
        t0 = perf_counter()
        pred_cloud, pred_feats = load_predicted_features(
            config.evaluation.predictions_path,
            config.evaluation.voxel_downsampling_size,
        )
        logger.info(f"Loaded predicted features in {perf_counter() - t0:.3f}s")

        # Load prompt list
        t0 = perf_counter()
        prompt_list = load_prompt_list(config)
        logger.info(f"Loaded prompt list in {perf_counter() - t0:.3f}s (count={len(prompt_list)})")

        # Evaluate predicted features
        t0 = perf_counter()
        logits = compute_feature_to_prompt_similarity(
            model=model,
            features=pred_feats,
            prompt_list=prompt_list,
        )
        logger.info(f"Computed feature-prompt similarities in {perf_counter() - t0:.3f}s")

        # Get predicted label from logits
        t0 = perf_counter()
        pred_labels = get_label_from_logits(
            logits, prompt_list, method="topn", topn=config.evaluation.topn
        )
        logger.info(f"Converted logits to labels in {perf_counter() - t0:.3f}s")

        results = {}
        pred_categories = None
        point_labels = None
        point_categories = None
        # Compute metric (intersection over union)
        if config.evaluation.freq:
            t0 = perf_counter()
            freq_results, pred_categories, point_labels, point_categories = (
                metric.category_frequency_topn(
                    pred_cloud=pred_cloud,
                    pred_labels=pred_labels,
                    gt_cloud=gt_cloud,
                    gt_ids=gt_ids,
                    gt_labels_handler=openlex3d_gt_handler,
                    excluded_labels=config.evaluation.excluded_labels,
                )
            )
            results["freq"] = freq_results
            logger.info(f"Computed frequency metric in {perf_counter() - t0:.3f}s")

        if config.evaluation.set_ranking:
            t0 = perf_counter()
            set_ranking_results = metric.set_based_ranking(
                pred_cloud=pred_cloud,
                gt_cloud=gt_cloud,
                gt_ids=gt_ids,
                gt_labels_handler=openlex3d_gt_handler,
                excluded_labels=config.evaluation.excluded_labels,
                logits=logits,
                prompt_list=prompt_list,
            )
            results["ranking"] = set_ranking_results
            logger.info(f"Computed set-ranking metric in {perf_counter() - t0:.3f}s")

        # Log results
        logger.info(f"Scene Metrics: {results}")

        # Export predicted clouds
        t0 = perf_counter()
        save_results(
            output_path=config.evaluation.output_path,
            dataset=config.dataset.name,
            scene=config.dataset.scene,
            algorithm=Path(
                config.evaluation.algorithm, f"top_{config.evaluation.topn}"
            ),
            reference_cloud=gt_cloud,
            pred_categories=pred_categories,
            results=results,
            point_labels=point_labels,
            point_categories=point_categories,
        )
        logger.info(f"Saved evaluation results in {perf_counter() - t0:.3f}s")
        # log results saved to
        logger.info(
            f"Results saved to {config.evaluation.output_path}"
            f"/{config.evaluation.algorithm}/top_{config.evaluation.topn}"
            f"/{config.dataset.name}/{config.dataset.scene}"
        )
        logger.info(f"Total evaluation time: {perf_counter() - overall_start:.3f}s")

    elif config.evaluation.type == "caption":
        raise NotImplementedError(f"{config.evaluation.type} not supported yet")
    else:
        raise NotImplementedError(f"{config.evaluation.type} not supported")


if __name__ == "__main__":
    main()
