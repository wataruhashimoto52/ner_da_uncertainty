
import json
import logging
import os
import time
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedModel
from transformers.trainer_pt_utils import nested_concat

from .data_utils import prepare_input
from .metrics_auprc import (
    compute_auprc_ner_flatten,
    compute_auprc_ner_flatten_span,
    compute_auprc_ner_flatten_without_other,
    compute_auprc_ner_sentence,
    compute_auprc_ner_sentence_span,
    compute_auprc_ner_sentence_without_other,
)

from .metrics_calibration_errors import (
    compute_calibration_error_ner_flatten,
    compute_calibration_error_ner_flatten_span,
    compute_calibration_error_ner_flatten_without_other,
    compute_calibration_error_ner_sentence,
    compute_calibration_error_ner_sentence_span,
    compute_calibration_error_ner_sentence_without_other,
    save_sentence_span_probabilities,
)
from .schemas import EvalPredictionV2, TokenClassifierOutputConf
from .train_utils import apply_dropout

logger = logging.getLogger(__name__)


def nested_numpify(tensors: torch.Tensor):
    "Numpify `tensors` (even if it's a nested list/tuple of tensors)."
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_numpify(t) for t in tensors)
    t = tensors.cpu()
    if t.dtype == torch.bfloat16:
        # As of Numpy 1.21.4, NumPy does not support bfloat16 (see
        # https://github.com/numpy/numpy/blob/a47ecdea856986cd60eabbd53265c2ca5916ad5d/doc/source/user/basics.types.rst ).
        # Until Numpy adds bfloat16, we must convert float32.
        t = t.to(torch.float32)

    if tensors.requires_grad:
        return t.detach().numpy()
    return t.numpy()


def align_predictions(
    predictions: np.ndarray,
    label_ids: np.ndarray,
    label_map: dict,
    tags: Optional[list[list[int]]] = None,
) -> tuple[list[int], list[int]]:

    if tags:
        preds = np.asarray(tags)
        batch_size = preds.shape[0]

        out_label_list = [[] for _ in range(batch_size)]
        preds_list = [[] for _ in range(batch_size)]

        for i in range(batch_size):
            for j in range(label_ids.shape[1]):
                if label_ids[i, j] != nn.CrossEntropyLoss().ignore_index:
                    out_label_list[i].append(label_map[label_ids[i][j]])
            preds_list[i].extend([label_map[tag] for tag in tags[i]])

    else:
        preds = np.argmax(predictions, axis=2)
        batch_size, seq_len = preds.shape

        out_label_list = [[] for _ in range(batch_size)]
        preds_list = [[] for _ in range(batch_size)]

        for i in range(batch_size):
            for j in range(seq_len):
                if label_ids[i, j] != nn.CrossEntropyLoss().ignore_index:
                    out_label_list[i].append(label_map[label_ids[i][j]])
                    preds_list[i].append(label_map[preds[i][j]])

    return preds_list, out_label_list


def make_eval_prediction(
    model: Union[PreTrainedModel, list[PreTrainedModel]],
    dataloader: DataLoader,
    output_path: str,
    seed: int,
    label_map: dict,
    split: str = "test",
    calibration_algorithm: Optional[str] = None,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    num_monte_carlo: int = 20,
) -> EvalPredictionV2:
    all_losses = None
    all_preds = None
    all_labels = None
    all_attentions = None
    all_input_ids = None
    all_tags = []

    tokenizer = AutoTokenizer.from_pretrained(model.name_or_path)

    start = 0
    model.eval()
    batches = tqdm(dataloader)

    start = time.time()
    for step, data in enumerate(batches):
        if device:
            data = prepare_input(data, device)
        if "MCDropout" in model.__class__.__name__:
            # model.train()
            model.apply(apply_dropout)
            confidences = []
            loss = []
            tag = []
            for i in range(num_monte_carlo):
                with torch.inference_mode():
                    output: TokenClassifierOutputConf = model(**data)
                confidences.append(output.confidences.cpu())
                loss.append(output.losses.cpu())
                tag.append(output.tags)

            stacked_confidences = torch.stack(confidences, dim=-1)
            predictions = nested_numpify(stacked_confidences.mean(-1))

            stacked_loss = torch.stack(loss, dim=-1)
            losses = stacked_loss.mean(-1)
            tags = output.tags  # Optional: CRF
            attentions = output.attentions

        else:
            with torch.inference_mode():
                output: TokenClassifierOutputConf = model(**data)

            predictions = nested_numpify(output.confidences)
            losses = output.losses
            tags = output.tags
            attentions = output.attentions

        labels = nested_numpify(data["labels"])

        all_input_ids = (
            data["input_ids"]
            if all_input_ids is None
            else nested_concat(
                all_input_ids,
                data["input_ids"],
                padding_index=tokenizer.pad_token_id,
            )
        )

        all_preds = (
            predictions
            if all_preds is None
            else nested_concat(all_preds, predictions, padding_index=-100)
        )
        all_labels = (
            labels
            if all_labels is None
            else nested_concat(all_labels, labels, padding_index=-100)
        )

        all_attentions = (
            attentions
            if all_attentions is None
            else nested_concat(all_attentions, attentions, padding_index=0)
        )

        if attentions is not None:
            attentions = nested_numpify(output.attentions)
            all_attentions = (
                attentions
                if all_attentions is None
                else nested_concat(all_attentions, attentions, padding_index=0)
            )

        if losses is not None:
            losses = nested_numpify(output.losses)
            all_losses = (
                losses
                if all_losses is None
                else nested_concat(all_losses, losses, padding_index=-100)
            )

        if tags is not None:
            all_tags.extend(output.tags)
    end = time.time()

    inference_time = str(end - start)
    logger.info("Inference time: " + inference_time)

    if split == "test":
        records = []
        for j in range(all_preds.shape[0]):
            tokens = tokenizer.convert_ids_to_tokens(
                all_input_ids[j].tolist(), skip_special_tokens=True
            )
            confs = all_preds[j].max(-1)
            preds = all_preds[j].argmax(-1)
            labs = all_labels[j]
            filtered_preds = [
                label_map[int(p)] for p, l in zip(preds, labs) if l != -100
            ]
            filtered_confs = [c.tolist() for c, l in zip(confs, labs) if l != -100]
            filtered_labs = [label_map[int(l)] for l in labs if l != -100]

            records.append(
                {
                    "text": " ".join(
                        [
                            comp
                            for t in tokens
                            if (comp := t.replace("▁", "")) and comp != ""
                        ]
                    ),
                    "confidences": filtered_confs,
                    "predictions": filtered_preds,
                    "labels": filtered_labs,
                }
            )

        with open(
            os.path.join(output_path, f"test_predictions_{str(seed)}.json"), "w"
        ) as f:
            json.dump(records, f, ensure_ascii=False, indent=4)

    return EvalPredictionV2(
        predictions=all_preds,
        label_ids=all_labels,
        losses=all_losses,
        tags=all_tags,
        attentions=all_attentions,
    )


def compute_metrics(
    evalprediction: EvalPredictionV2,
    label_map: dict,
    output_path: str,
    split: str = "test",
    seed: int = 1,
) -> dict[str, float]:
    preds_list, out_label_list = align_predictions(
        evalprediction.predictions,
        evalprediction.label_ids,
        label_map=label_map,
        tags=evalprediction.tags,
    )

    metrics = {
        "accuracy_score": accuracy_score(out_label_list, preds_list),
        "precision": precision_score(out_label_list, preds_list),
        "recall": recall_score(out_label_list, preds_list),
        "f1": f1_score(out_label_list, preds_list),
        "expected_calibration_error_sentence": compute_calibration_error_ner_sentence(
            evalprediction,
            "ece",
        ),
        "expected_calibration_error_sentence_without_other": compute_calibration_error_ner_sentence_without_other(
            evalprediction, label_map, "ece"
        ),
        "expected_calibration_error_sentence_span": compute_calibration_error_ner_sentence_span(
            evalprediction,
            label_map,
            False,
            "ece",
        ),
        "expected_calibration_error_flatten": compute_calibration_error_ner_flatten(
            evalprediction,
            "ece",
        ),
        "expected_calibration_error_flatten_without_other": compute_calibration_error_ner_flatten_without_other(
            evalprediction, label_map, "ece"
        ),
        "expected_calibration_error_flatten_span": compute_calibration_error_ner_flatten_span(
            evalprediction,
            label_map,
            False,
            "ece",
        ),
        "maximized_calibration_error_sentence": compute_calibration_error_ner_sentence(
            evalprediction,
            "mce",
        ),
        "maximized_calibration_error_sentence_without_other": compute_calibration_error_ner_sentence_without_other(
            evalprediction, label_map, "mce"
        ),
        "maximized_calibration_error_sentence_span": compute_calibration_error_ner_sentence_span(
            evalprediction,
            label_map,
            False,
            "mce",
        ),
        "maximized_calibration_error_flatten": compute_calibration_error_ner_flatten(
            evalprediction,
            "mce",
        ),
        "maximized_calibration_error_flatten_without_other": compute_calibration_error_ner_flatten_without_other(
            evalprediction, label_map, "mce"
        ),
        "maximized_calibration_error_flatten_span": compute_calibration_error_ner_flatten_span(
            evalprediction,
            label_map,
            False,
            "mce",
        ),
        "auprc_sentence_macro": compute_auprc_ner_sentence(
            evalprediction,
            label_map,
            average="macro",
        ),
        "auprc_sentence_weighted": compute_auprc_ner_sentence(
            evalprediction,
            label_map,
            average="weighted",
        ),
        "auprc_sentence_without_other_macro": compute_auprc_ner_sentence_without_other(
            evalprediction,
            label_map,
            average="macro",
        ),
        "auprc_sentence_without_other_weighted": compute_auprc_ner_sentence_without_other(
            evalprediction,
            label_map,
            average="weighted",
        ),
        "auprc_sentence_span": compute_auprc_ner_sentence_span(
            evalprediction,
            label_map,
            use_mean=False,
        ),
        "auprc_flatten_macro": compute_auprc_ner_flatten(
            evalprediction,
            average="macro",
        ),
        "auprc_flatten_weighted": compute_auprc_ner_flatten(
            evalprediction,
            average="weighted",
        ),
        "auprc_flatten_without_other_macro": compute_auprc_ner_flatten_without_other(
            evalprediction,
            label_map,
            average="macro",
        ),
        "auprc_flatten_without_other_weighted": compute_auprc_ner_flatten_without_other(
            evalprediction,
            label_map,
            average="weighted",
        ),
        "auprc_flatten_span": compute_auprc_ner_flatten_span(
            evalprediction,
            label_map,
            ignore_index=None,
            use_mean=False,
        )
    }

    if evalprediction.losses is not None:
        metrics.update({"loss": np.asarray(evalprediction.losses).sum()})

    # save span probabilities
    if split == "test":
        save_sentence_span_probabilities(
            evalprediction,
            os.path.join(output_path, f"{split}_span_probs_labels_{str(seed)}.pkl"),
            label_map,
        )

    return metrics


def evaluation(
    steps: Optional[int],
    model: Union[PreTrainedModel, list[PreTrainedModel]],
    dataloader: DataLoader,
    label_map: dict,
    output_path: str,
    calibration_algorithm: Optional[str] = None,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    writer: Optional[SummaryWriter] = None,
    num_monte_carlo: int = 20,
    split: str = "test",
    seed: int = 1,
) -> dict[str, float]:
    logger.info("=== make eval prediction ===")
    evalprediction = make_eval_prediction(
        model,
        dataloader,
        output_path,
        seed,
        split,
        label_map,
        calibration_algorithm,
        device,
        num_monte_carlo,
    )
    logger.info("=== computing metrics ===")
    metrics = compute_metrics(evalprediction, label_map, output_path, split, seed)

    return metrics
