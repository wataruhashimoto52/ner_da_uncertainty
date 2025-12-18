from typing import Optional

import numpy as np
import torch
from seqeval.metrics.sequence_labeling import get_entities
from torchmetrics.functional.classification import (
    binary_average_precision,
)

from .schemas import EvalPredictionV2


# SeqEval as well as transformers use -100 to mask padded tokens.
IGNORE_INDEX = -100


def _find_o_tag(label_map: dict) -> int:
    try:
        return next(k for k, v in label_map.items() if v == "O")
    except StopIteration as exc:
        raise ValueError("Label map does not contain the 'O' tag.") from exc


def _token_uncertainty_and_errors(
    confidences: np.ndarray,
    labels: np.ndarray,
):
    mask = labels != IGNORE_INDEX
    if not np.any(mask):
        return (
            np.asarray([]),
            np.asarray([]),
            np.asarray([]),
            np.asarray([]),
        )

    filtered_confidences = confidences[mask]
    filtered_labels = labels[mask]
    predicted_labels = np.argmax(filtered_confidences, axis=-1)
    max_confidences = filtered_confidences[
        np.arange(filtered_confidences.shape[0]),
        predicted_labels,
    ]
    scores = 1.0 - max_confidences
    errors = (predicted_labels != filtered_labels).astype(np.int64)

    return scores, errors, predicted_labels, filtered_labels


def _binary_auprc_from_scores(
    scores: np.ndarray,
    targets: np.ndarray,
    ignore_index: Optional[int],
) -> Optional[float]:
    if scores.size == 0:
        return None

    preds_tensor = torch.as_tensor(scores, dtype=torch.float32)
    target_tensor = torch.as_tensor(targets, dtype=torch.int64)
    score = binary_average_precision(
        preds=preds_tensor,
        target=target_tensor,
        ignore_index=ignore_index,
    )
    if torch.isnan(score):
        return None
    return float(score)


def compute_auprc_ner_flatten(
    evalprediction: EvalPredictionV2,
    ignore_index: Optional[int] = None,
    average: str = "macro",
) -> float:
    confidences = evalprediction.predictions

    base_confidences = np.reshape(
        confidences, (-1, confidences.shape[-1])
    )  # (length, num_labels)
    base_labels = evalprediction.label_ids.flatten()  # (length, )
    final_confidences = np.asarray(
        [p for (p, l) in zip(base_confidences, base_labels) if l != IGNORE_INDEX]
    )

    final_labels = np.asarray([l for l in base_labels if l != IGNORE_INDEX])

    scores, errors, _, _ = _token_uncertainty_and_errors(
        final_confidences,
        final_labels,
    )

    ap = _binary_auprc_from_scores(scores, errors, ignore_index=ignore_index)
    if ap is None:
        return 0.0
    return ap


def compute_auprc_ner_flatten_without_other(
    evalprediction: EvalPredictionV2,
    label_map: dict,
    ignore_index: Optional[int] = None,
    average: str = "macro",
) -> Optional[float]:

    confidences = evalprediction.predictions

    base_confidences = np.reshape(
        confidences, (-1, confidences.shape[-1])
    )  # (length, num_labels)
    base_labels = evalprediction.label_ids.flatten()  # (length, )

    processed_confidences = np.asarray(
        [p for (p, l) in zip(base_confidences, base_labels) if l != IGNORE_INDEX]
    )

    processed_labels = np.asarray([l for l in base_labels if l != IGNORE_INDEX])

    scores, errors, predicted_labels, _ = _token_uncertainty_and_errors(
        processed_confidences,
        processed_labels,
    )

    if scores.size == 0:
        return None

    o_tag = _find_o_tag(label_map)
    mask = predicted_labels != o_tag
    if not np.any(mask):
        return None

    filtered_scores = scores[mask]
    filtered_errors = errors[mask]

    return _binary_auprc_from_scores(
        filtered_scores,
        filtered_errors,
        ignore_index=ignore_index,
    )


def compute_auprc_ner_flatten_span(
    evalprediction: EvalPredictionV2,
    label_map: dict,
    ignore_index: Optional[int] = None,
    use_mean: bool = False,
) -> Optional[float]:

    confidences = evalprediction.predictions

    base_confidences = np.reshape(
        confidences, (-1, confidences.shape[-1])
    )  # (length, num_labels)
    base_labels = evalprediction.label_ids.flatten()  # (length, )

    processed_confidences = np.asarray(
        [p for (p, l) in zip(base_confidences, base_labels) if l != IGNORE_INDEX]
    )

    processed_labels = np.asarray([l for l in base_labels if l != IGNORE_INDEX])
    processed_label_entities = get_entities([label_map[l] for l in processed_labels])

    argmax_prediction = np.argmax(processed_confidences, 1)
    processed_predicted_entities = get_entities(
        [label_map[p] for p in argmax_prediction]
    )

    assert len(processed_confidences) == len(processed_labels)

    preds = []
    trues = []
    for i in range(len(processed_confidences)):
        e_true = [ent for ent in processed_label_entities if ent[1] == i]
        e_pred = [ent for ent in processed_predicted_entities if ent[1] == i]

        if not e_true and not e_pred:
            continue
        if e_pred:
            if use_mean:
                conf = np.mean(
                    np.max(processed_confidences[e_pred[0][1] : e_pred[0][2] + 1, :], 1)
                )
            else:
                conf = np.prod(
                    np.max(processed_confidences[e_pred[0][1] : e_pred[0][2] + 1, :], 1)
                )
            error_score = 1.0 - conf
            if not e_true or e_true[0] != e_pred[0]:
                preds.append(error_score)
                trues.append(1)
            else:
                preds.append(error_score)
                trues.append(0)

        else:  # not e_pred but e_true
            preds.append(1.0)
            trues.append(1)

    return _binary_auprc_from_scores(
        np.asarray(preds),
        np.asarray(trues),
        ignore_index=ignore_index,
    )


def compute_auprc_ner_sentence(
    evalprediction: EvalPredictionV2,
    label_map: dict,
    ignore_index: Optional[int] = None,
    average: str = "macro",
) -> Optional[float]:
    confidences = evalprediction.predictions
    labels = evalprediction.label_ids
    attentions = evalprediction.attentions

    auprcs = []
    weights = []
    for i in range(len(confidences)):
        confidence = confidences[i]  # (length, num_labels)
        label = labels[i]  # (length, )
        final_confidences = np.asarray(
            [p for (p, l) in zip(confidence, label) if l != IGNORE_INDEX]
        )

        final_labels = np.asarray([l for l in label if l != IGNORE_INDEX])

        scores, errors, _, _ = _token_uncertainty_and_errors(
            final_confidences,
            final_labels,
        )

        if scores.size == 0:
            continue

        score = _binary_auprc_from_scores(
            scores,
            errors,
            ignore_index=ignore_index,
        )
        if score is None:
            continue
        auprcs.append(score)
        weights.append(len(scores))

    if not auprcs:
        return None

    if average == "weighted" and weights:
        return float(np.average(auprcs, weights=weights))
    return float(np.mean(auprcs))


def compute_auprc_ner_sentence_without_other(
    evalprediction: EvalPredictionV2,
    label_map: dict,
    ignore_index: Optional[int] = None,
    average: str = "macro",
) -> Optional[float]:
    confidences = evalprediction.predictions
    labels = evalprediction.label_ids
    attentions = evalprediction.attentions

    o_tag = _find_o_tag(label_map)

    auprcs = []
    weights = []
    for i in range(len(confidences)):
        confidence = confidences[i]  # (length, num_labels)
        label = labels[i]  # (length, )

        processed_confidences = np.asarray(
            [p for (p, l) in zip(confidence, label) if l != IGNORE_INDEX]
        )

        processed_labels = np.asarray([l for l in label if l != IGNORE_INDEX])

        scores, errors, predicted_labels, _ = _token_uncertainty_and_errors(
            processed_confidences,
            processed_labels,
        )

        if scores.size == 0:
            continue

        mask = predicted_labels != o_tag
        if not np.any(mask):
            continue

        filtered_scores = scores[mask]
        filtered_errors = errors[mask]

        score = _binary_auprc_from_scores(
            filtered_scores,
            filtered_errors,
            ignore_index=ignore_index,
        )
        if score is None:
            continue

        auprcs.append(score)
        weights.append(len(filtered_scores))

    if not auprcs:
        return None

    if average == "weighted" and weights:
        return float(np.average(auprcs, weights=weights))
    return float(np.mean(auprcs))


def compute_auprc_ner_sentence_span(
    evalprediction: EvalPredictionV2,
    label_map: dict,
    ignore_index: Optional[int] = None,
    use_mean: bool = False,
) -> Optional[float]:

    confidences = evalprediction.predictions  # (N, seq_length, confidence)
    labels = evalprediction.label_ids  # (N, seq_length, )

    auprcs = []
    for i in range(len(confidences)):
        processed_confidences = np.asarray(
            [p for (p, l) in zip(confidences[i], labels[i]) if l != IGNORE_INDEX]
        )

        processed_labels = np.asarray([l for l in labels[i] if l != IGNORE_INDEX])
        processed_label_entities = get_entities(
            [label_map[l] for l in processed_labels]
        )

        argmax_prediction = np.argmax(processed_confidences, 1)
        processed_predicted_entities = get_entities(
            [label_map[p] for p in argmax_prediction]
        )

        assert len(processed_confidences) == len(processed_labels)

        preds = []
        trues = []
        for i in range(len(processed_confidences)):
            e_true = [ent for ent in processed_label_entities if ent[1] == i]
            e_pred = [ent for ent in processed_predicted_entities if ent[1] == i]

            if not e_true and not e_pred:
                continue
            if e_pred:
                if use_mean:
                    conf = np.mean(
                        np.max(
                            processed_confidences[e_pred[0][1] : e_pred[0][2] + 1, :], 1
                        )
                    )
                else:
                    conf = np.prod(
                        np.max(
                            processed_confidences[e_pred[0][1] : e_pred[0][2] + 1, :], 1
                        )
                    )
                error_score = 1.0 - conf
                if not e_true or e_true[0] != e_pred[0]:
                    preds.append(error_score)
                    trues.append(1)
                else:
                    preds.append(error_score)
                    trues.append(0)

            else:  # not e_pred but e_true
                preds.append(1.0)
                trues.append(1)

        score = _binary_auprc_from_scores(
            np.asarray(preds),
            np.asarray(trues),
            ignore_index=ignore_index,
        )
        if score is None:
            continue
        auprcs.append(score)

    if not auprcs:
        return None

    return float(np.mean(auprcs))
