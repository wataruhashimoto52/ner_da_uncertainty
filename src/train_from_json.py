import json
import logging
import math
import os
import re
import sys
from dataclasses import dataclass, field
from functools import partial
from importlib import import_module
from typing import Dict, Optional, Union

import torch
from datasets import load_dataset
from torch import nn
from tqdm import tqdm
from transformers import (
    AdamW,
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    TrainingArguments,
    get_linear_schedule_with_warmup,
    set_seed,
)
from utils.data_utils import CustomDataCollatorForTokenClassification, prepare_input
from utils.evaluations import evaluation
from utils.schemas import TokenClassifierOutputConf
from utils.tasks import NER
from utils.tokenization_utils import tokenize_and_align_labels_from_json
from utils.train_utils import EarlyStoppingForTransformers

logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    model_type: str = field(
        default="AutoModelForTokenClassification",
        metadata={"help": f"Model type selected"},
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    task_type: Optional[str] = field(
        default="NER",
        metadata={"help": "Task type to fine tune in training (e.g. NER, POS, etc)"},
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    use_fast: bool = field(
        default=False, metadata={"help": "Set this flag to use fast tokenization."}
    )
    # If you want to tweak more attributes on your tokenizer, you should do it in a distinct script,
    # or just modify its tokenizer_config.json.
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from s3"
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    data_path: str = field(default=None, metadata={"help": "data path"})
    test_data_path: str = field(default=None, metadata={"help": "test data path"})
    labels: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to a file containing all labels. If not specified, CoNLL-2003 labels are used."
        },
    )
    max_seq_length: int = field(
        default=None,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to model maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    classifier_lr: float = field(
        default=5e-5, metadata={"help": "learning rate for classifier"}
    )
    calibration_algorithm: str = field(
        default=None,
        metadata={"help": "Choose your calibration algorithm. default is None."},
    )
    tau: float = field(
        default=None,
        metadata={"help": "temperature parameter."},
    )
    run_multiple_seeds: bool = field(
        default=False,
        metadata={"help": "multiple runs with different seeds. seed range is [1, 10]"},
    )
    num_monte_carlo: int = field(
        default=20, metadata={"help": "Number of Monte Carlo approximation samples."}
    )

    smoothing: float = field(
        default=0.1,
        metadata={
            "help": "smoothing rate",
        },
    )



def main(
    model_args: ModelArguments,
    data_args: DataTrainingArguments,
    training_args: TrainingArguments,
    seed: Optional[int] = None,
):
    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. "
            "Use --overwrite_output_dir to overcome."
        )

    os.makedirs(name=training_args.output_dir, exist_ok=True)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    if seed is None:
        seed = training_args.seed
        set_seed(seed)
    else:
        set_seed(seed)

    # Get datasets
    token_classification_task = NER()
    labels = token_classification_task.get_labels(data_args.labels)
    label_map: Dict[int, str] = {i: label for i, label in enumerate(labels)}
    label2id: Dict[str, int] = {label: i for i, label in enumerate(labels)}
    num_labels = len(labels)

    padding = "max_length" if data_args.pad_to_max_length else False

    # prepare model settings
    config: PretrainedConfig = AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name
        else model_args.model_name_or_path,
        num_labels=num_labels,
        id2label=label_map,
        label2id=label2id,
        cache_dir=model_args.cache_dir,
    )
    tokenizer: Union[
        PreTrainedTokenizer, PreTrainedTokenizerFast
    ] = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name
        else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = import_module("models")
    try:
        auto_model = getattr(module, model_args.model_type)
        if re.search(r"TemperatureScaled", model_args.model_type):
            model: PreTrainedModel = auto_model.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                tau=data_args.tau,
            )
        elif re.search(
            r"LabelSmoothing",
            model_args.model_type,
        ):
            model: PreTrainedModel = auto_model.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                smoothing=data_args.smoothing,
            )
        else:
            model: PreTrainedModel = auto_model.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
            )

    except AttributeError:
        raise ValueError(
            f"{model_args.model_type} is not defined."
        )
    tokenization_partial_func = partial(
        tokenize_and_align_labels_from_json,
        tokenizer=tokenizer,
        padding=padding,
        label2id=label2id,
    )
    base_dataset = load_dataset(
        "json",
        data_files={
            "train": os.path.join(data_args.data_path, "train.json"),
            "validation": os.path.join(data_args.data_path, "dev.json"),
        },
    )
    base_dataset = base_dataset.shuffle(seed=seed)

    base_test_dataset = load_dataset(
        "json", data_files={"test": os.path.join(data_args.test_data_path, "test.json")}
    ).shuffle(seed=seed)

    dataset = base_dataset.map(tokenization_partial_func, batched=True)
    test_dataset = base_test_dataset.map(tokenization_partial_func, batched=True)

    # Data collator
    data_collator = CustomDataCollatorForTokenClassification(
        tokenizer,
        id2label=config.id2label,
        label2id=config.label2id,
        max_length=data_args.max_seq_length,
        padding=True,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )
    columns_to_remove = ["id", "tokens", "ner_tags"]
    if re.search(r"multiconer", data_args.data_path):
        columns_to_remove = ["domain", "tokens", "ner_tags"]

    dataset = dataset.remove_columns(columns_to_remove)
    test_dataset = test_dataset.remove_columns(columns_to_remove)

    # dataset.set_format("torch")
    train_dataloader = torch.utils.data.DataLoader(
        dataset["train"],
        shuffle=True,
        collate_fn=data_collator,
        batch_size=training_args.per_gpu_train_batch_size,
        drop_last=training_args.dataloader_drop_last,
        pin_memory=training_args.dataloader_pin_memory,
    )
    eval_dataloader = torch.utils.data.DataLoader(
        dataset["validation"],
        shuffle=True,
        collate_fn=data_collator,
        batch_size=training_args.per_gpu_train_batch_size,
        drop_last=training_args.dataloader_drop_last,
        pin_memory=training_args.dataloader_pin_memory,
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset["test"],
        shuffle=True,
        collate_fn=data_collator,
        batch_size=training_args.per_gpu_train_batch_size,
        drop_last=training_args.dataloader_drop_last,
        pin_memory=training_args.dataloader_pin_memory,
    )
    model.to(device)
    model.gradient_checkpointing_enable()

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": training_args.weight_decay,
            "lr": training_args.learning_rate,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
            "lr": training_args.learning_rate,
        },
    ]

    optimizer = AdamW(
        params=optimizer_grouped_parameters,
        betas=(training_args.adam_beta1, training_args.adam_beta2),
        eps=training_args.adam_epsilon,
    )

    num_train_steps = math.ceil(len(train_dataloader) * training_args.num_train_epochs)
    if training_args.warmup_steps > 0:
        num_warmup_steps = training_args.warmup_steps
    elif training_args.warmup_ratio > 0:
        num_warmup_steps = int(num_train_steps * training_args.warmup_ratio)
    else:
        num_warmup_steps = 0
    num_warmup_steps = 0

    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_train_steps,
    )

    early_stopping = EarlyStoppingForTransformers(
        path=training_args.output_dir, patience=5, verbose=True
    )
    best_model: Optional[PreTrainedModel] = None

    for epoch in range(int(training_args.num_train_epochs)):
        batches = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch + 1}/{int(training_args.num_train_epochs)}",
        )
        model.train()
        for batch in batches:
            model.zero_grad()
            batch = prepare_input(batch, device)
            outputs: TokenClassifierOutputConf = model(**batch)

            # gradient clipping
            if hasattr(model, "clip_grad_norm_"):
                model.clip_grad_norm_(training_args.max_grad_norm)
            elif hasattr(optimizer, "clip_grad_norm"):
                optimizer.clip_grad_norm(training_args.max_grad_norm)
            else:
                nn.utils.clip_grad.clip_grad_norm_(
                    parameters=model.parameters(), max_norm=training_args.max_grad_norm
                )

            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            batches.set_postfix({"loss": loss.item()})

            model.zero_grad()

        logger.info("Evaluation")
        eval_results = evaluation(
            steps=epoch,
            model=model,
            dataloader=eval_dataloader,
            label_map=label_map,
            output_path=training_args.output_dir,
            calibration_algorithm=data_args.calibration_algorithm,
            device=device,
            writer=None,
            num_monte_carlo=data_args.num_monte_carlo,
            split="dev",
            seed=seed,
        )
        eval_results.update({"learning_rate": lr_scheduler.get_last_lr()[0]})
        is_best = early_stopping(eval_results["f1"], model, tokenizer, save_flag=False)
        if is_best:
            best_weight = model.state_dict()
            best_model = model.to(device)
            best_model.load_state_dict(best_weight)

        output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as f:
            logger.info("***** Eval results *****")
            for key, value in eval_results.items():
                logger.info("  %s = %s", key, value)
                f.write("%s = %s\n" % (key, value))

        if early_stopping.early_stop:
            logger.info("Early Stopped.")
            break

    # Test
    logger.info("Test")

    # load best model
    # model.to(device)
    test_results = evaluation(
        steps=None,
        model=best_model,
        dataloader=test_dataloader,
        label_map=label_map,
        output_path=training_args.output_dir,
        calibration_algorithm=data_args.calibration_algorithm,
        device=device,
        writer=None,
        num_monte_carlo=data_args.num_monte_carlo,
        split="test",
        seed=seed,
    )

    test_results.update({"learning_rate": lr_scheduler.get_last_lr()[0]})
    output_test_results_file = os.path.join(
        training_args.output_dir, f"test_results_{str(seed)}.txt"
    )
    with open(output_test_results_file, "w") as f:
        logger.info("***** Test results *****")
        for key, value in test_results.items():
            logger.info("  %s = %s", key, value)
            f.write("%s = %s\n" % (key, value))

    # if not Early Stopped, save the final model.
    if early_stopping.early_stop is False:
        # Save Model
        best_model.save_pretrained(training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if data_args.run_multiple_seeds:
        for i in range(1, 11):  # 1 ~ 10
            main(model_args, data_args, training_args, seed=i)
    else:
        main(model_args, data_args, training_args)
