from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn import CrossEntropyLoss
from transformers import (
    DebertaV2ForTokenClassification,
    DebertaV2Model,
    PretrainedConfig,
)

from algorithms.label_smoothing import LabelSmoother
from utils.schemas import TokenClassifierOutputConf


class BaselineDeBERTaV3ForTokenClassification(DebertaV2ForTokenClassification):
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        tokens: Optional[list[list[str]]] = None,
        ner_tags: Optional[list[list[str]]] = None,
        word_ids: Optional[list[list[int]]] = None,
        valid_masks: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        masks_for_span: Optional[torch.Tensor] = None,
        labels_for_span: Optional[torch.Tensor] = None,
        binary_labels_for_span: Optional[torch.Tensor] = None,
        token_masks: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = True,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutputConf]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.deberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        losses = None
        confidences = None
        span_logits = None
        if labels is not None:

            confidences = F.softmax(logits, dim=-1)
            logits_reshaped = rearrange(logits, "b l k -> (b l) k")
            labels_reshaped = rearrange(labels, "b l -> (b l)")

            loss_fct = CrossEntropyLoss(reduction="none")
            losses: torch.Tensor = loss_fct(logits_reshaped, labels_reshaped)

            loss_fnc_mean = CrossEntropyLoss(reduction="mean")
            loss: torch.Tensor = loss_fnc_mean(logits_reshaped, labels_reshaped)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutputConf(
            loss=loss,  # scalar
            logits=logits,  # (N, L, K)
            hidden_states=outputs.hidden_states,
            attentions=None,
            confidences=confidences,  # (N, L, K)
            losses=losses,  # (N, )
            span_logits=span_logits,
            sequence_output=sequence_output,
        )


class LabelSmoothingDeBERTaV3ForTokenClassification(
    BaselineDeBERTaV3ForTokenClassification
):
    def __init__(self, config: PretrainedConfig, smoothing: float) -> None:
        super().__init__(config)
        self.num_labels = config.num_labels
        self.deberta = DebertaV2Model(config)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.smoothing = smoothing

        self.criterion_none = LabelSmoother(reduction="none", epsilon=self.smoothing)
        self.criterion_mean = LabelSmoother(reduction="mean", epsilon=self.smoothing)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        tokens: Optional[list[list[str]]] = None,
        ner_tags: Optional[list[list[str]]] = None,
        word_ids: Optional[list[list[int]]] = None,
        valid_masks: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        masks_for_span: Optional[torch.Tensor] = None,
        labels_for_span: Optional[torch.Tensor] = None,
        binary_labels_for_span: Optional[torch.Tensor] = None,
        token_masks: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = True,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutputConf]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.deberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits: torch.Tensor = self.classifier(sequence_output)

        loss = None
        losses = None
        confidences = None
        span_logits = None
        if labels is not None:
            confidences = F.softmax(logits, dim=-1)
            reshaped_logits = rearrange(logits, "b l h -> (b l) h")
            reshaped_labels = rearrange(labels, "b l -> (b l)")
            losses: torch.Tensor = self.criterion_none(reshaped_logits, reshaped_labels)
            loss: torch.Tensor = self.criterion_mean(reshaped_logits, reshaped_labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutputConf(
            loss=loss,  # scalar
            logits=logits,  # (N, L, K)
            hidden_states=outputs.hidden_states,
            attentions=None,
            confidences=confidences,  # (N, L, K)
            losses=losses,  # (N, )
            span_logits=span_logits,
        )


class TemperatureScaledDeBERTaV3ForTokenClassification(
    BaselineDeBERTaV3ForTokenClassification
):
    def __init__(self, config: PretrainedConfig, tau: float) -> None:
        super().__init__(config)
        self.num_labels = config.num_labels
        self.deberta = DebertaV2Model(config)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.tau = tau

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        valid_masks: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        masks_for_span: Optional[torch.Tensor] = None,
        labels_for_span: Optional[torch.Tensor] = None,
        binary_labels_for_span: Optional[torch.Tensor] = None,
        token_masks: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = True,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutputConf]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.deberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        losses = None
        confidences = None
        span_logits = None
        if labels is not None:
            logits = logits / self.tau
            loss_fct = CrossEntropyLoss(reduction="none")
            confidences = F.softmax(logits, dim=-1)
            losses: torch.Tensor = loss_fct(
                logits.view(-1, self.num_labels), labels.view(-1)
            )

            loss_fnc_mean = CrossEntropyLoss()
            loss: torch.Tensor = loss_fnc_mean(
                logits.view(-1, self.num_labels), labels.view(-1)
            )

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutputConf(
            loss=loss,  # scalar
            logits=logits,  # (N, L, K)
            hidden_states=outputs.hidden_states,
            attentions=None,
            confidences=confidences,  # (N, L, K)
            losses=losses,  # (N, )
            span_logits=span_logits,
        )


class MCDropoutDeBERTaV3ForTokenClassification(DebertaV2ForTokenClassification):
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        valid_masks: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        masks_for_span: Optional[torch.Tensor] = None,
        labels_for_span: Optional[torch.Tensor] = None,
        binary_labels_for_span: Optional[torch.Tensor] = None,
        token_masks: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = True,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutputConf]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.deberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        losses = None
        confidences = None
        span_logits = None
        if labels is not None:

            loss_fct = CrossEntropyLoss(reduction="none")
            confidences = F.softmax(logits, dim=-1)
            losses: torch.Tensor = loss_fct(
                logits.view(-1, self.num_labels), labels.view(-1)
            )

            loss_fnc_mean = CrossEntropyLoss()
            loss: torch.Tensor = loss_fnc_mean(
                logits.view(-1, self.num_labels), labels.view(-1)
            )

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutputConf(
            loss=loss,  # scalar
            logits=logits,  # (N, L, K)
            hidden_states=outputs.hidden_states,
            attentions=None,
            confidences=confidences,  # (N, L, K)
            losses=losses,  # (N, )
            span_logits=span_logits,
        )


class LabelWiseReplacementDeBERTaV3ForTokenClassification(
    BaselineDeBERTaV3ForTokenClassification
):
    pass


class MentionReplacementDeBERTaV3ForTokenClassification(
    BaselineDeBERTaV3ForTokenClassification
):
    pass


class SynonymReplacementDeBERTaV3ForTokenClassification(
    BaselineDeBERTaV3ForTokenClassification
):
    pass
