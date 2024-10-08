"""
ref. 

https://github.com/RandyZhouRan/MELM 
"""

import json
import os
import random
import time
import re
from typing import Dict, Union

import numpy as np
import torch
import torch.optim as optim
from datasets import Dataset
from tap import Tap
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from algorithms.melm_data_for_trainer import DataForTrainer
from algorithms.melm_data_gen import Data
from utils.tasks import NER

device = torch.device("cuda")


class MELMArguments(Tap):
    batch_size: int = 16
    lr: float = 1e-5
    grad_acc: int = 2
    n_epochs: int = 20
    clip: float = 1.0
    sigma: float = 1.0
    o_mask_rate: float = 0.0
    k: int = 5
    sub_idx: int = -1
    base_model_name_or_path: str  # microsoft/mdeberta-v3-base
    mu_ratio: float
    mask_rate: float
    label_path: str  # "data/ner_data/conll2003/labels.txt"
    data_path: str  # "data/ner_data/conll2003/"
    output_path: str  # "models/melm_outputs/"
    seed: int

    def configure(self) -> None:
        self.add_argument("--base_model_name_or_path", type=str, required=True)
        self.add_argument("--mu_ratio", type=float, required=True)
        self.add_argument("--mask_rate", type=float, required=True)
        self.add_argument("--label_path", type=str, required=True)
        self.add_argument("--data_path", type=str, required=True)
        self.add_argument("--output_path", type=str, required=True)
        self.add_argument("--seed", type=int, required=True)


def train(model, iterator, optimizer, clip, grad_acc, epoch):

    model.train()
    train_start = time.time()

    optimizer.zero_grad()

    for i, batch in tqdm(enumerate(iterator), total=len(iterator)):
        batch_start = time.time()
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, label_ids, masked_ids, entity_mask = batch
        # Different masking for diff epoch
        epoch_remainder = epoch % 30
        masked_ids = masked_ids[:, epoch_remainder]
        entity_mask = entity_mask[:, epoch_remainder]

        batch_size = label_ids.shape[0]
        outputs = model(
            masked_ids, input_mask, labels=input_ids, output_hidden_states=True
        )
        loss = outputs.loss
        logits = outputs.logits
        last_hids = outputs.hidden_states[-1]
        embs = outputs.hidden_states[0]

        loss = loss / grad_acc

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        if (i + 1) % grad_acc == 0:
            optimizer.step()
            optimizer.zero_grad()


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def evaluate(model, iterator):
    model.eval()
    with torch.no_grad():

        epoch_loss = 0
        correct_count = 0
        total_count = 0
        entity_correct = 0
        entity_total = 0

        for i, batch in enumerate(iterator):
            batch_start = time.time()
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, label_ids, masked_ids, entity_mask = batch

            # Use first masking for evaluation
            masked_ids = masked_ids[:, 0]
            entity_mask = entity_mask[:, 0]

            batch_size = label_ids.shape[0]

            outputs = model(masked_ids, input_mask, labels=input_ids)
            loss = outputs.loss
            logits = outputs.logits

            epoch_loss += loss

            pred = torch.argmax(logits, dim=-1)

            match = (input_ids == pred) * input_mask
            correct_count += torch.sum(match).item()
            total_count += torch.sum(input_mask).item()

            entity_match = (input_ids == pred) * entity_mask
            entity_correct += torch.sum(entity_match).item()
            entity_total += torch.sum(entity_mask).item()

    return (
        epoch_loss / (i + 1),
        correct_count / total_count,
        entity_correct / entity_total,
    )


def aug(entity_model, o_model, iterator, k, sub_idx):

    print("Augmenting sentences with MELM checkpoint ...")

    assert sub_idx <= k
    entity_model.eval()
    o_model.eval()

    batches_of_entity_aug = []
    batches_of_o_aug = []
    batches_of_total_aug = []  # Combine both entity aug and O aug
    batches_of_input = []

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            batch_start = time.time()
            batch = tuple(t.to(device) for t in batch)
            (
                input_ids,
                input_mask,
                label_ids,
                entity_masked_ids,
                entity_mask,
                o_masked_ids,
                o_mask,
            ) = batch

            # Entity aug
            entity_outputs = entity_model(
                entity_masked_ids, input_mask, labels=input_ids
            )
            entity_logits = entity_outputs.logits

            top_logits, topk = torch.topk(entity_logits, k=k, dim=-1)
            # torch. set_printoptions(profile="full")
            # print("indices \n", topk[:,:30,:5])
            if sub_idx != -1:
                sub = topk[:, :, sub_idx]
            else:  # random picking from topk
                gather_idx = torch.randint(1, k, (topk.shape[0], topk.shape[1], 1)).to(
                    device
                )
                sub = torch.gather(topk, -1, gather_idx).squeeze(-1)

            entity_aug = torch.where(entity_mask == 1, sub, entity_masked_ids)
            batches_of_entity_aug.append(entity_aug)

            # O aug
            o_outputs = o_model(o_masked_ids, input_mask, labels=input_ids)
            o_logits = o_outputs.logits

            top_logits, topk = torch.topk(o_logits, k=k, dim=-1)
            sub = topk[:, :, 0]  # Always use top prediction for O-masks
            o_aug = torch.where(o_mask == 1, sub, o_masked_ids)
            batches_of_o_aug.append(o_aug)

            # Combine both entity aug and O aug
            total_aug = torch.where(o_mask == 1, sub, entity_aug)
            batches_of_total_aug.append(total_aug)

            batches_of_input.append(input_ids)

    entity_aug_tensor = torch.cat(batches_of_entity_aug, dim=0)
    o_aug_tensor = torch.cat(batches_of_o_aug, dim=0)
    total_aug_tensor = torch.cat(batches_of_total_aug, dim=0)
    input_tensor = torch.cat(batches_of_input, dim=0)

    assert entity_aug_tensor.shape == input_tensor.shape
    assert o_aug_tensor.shape == input_tensor.shape
    assert total_aug_tensor.shape == input_tensor.shape

    return entity_aug_tensor, o_aug_tensor, total_aug_tensor, input_tensor


def decode(
    entity_aug_tensor,
    o_aug_tensor,
    total_aug_tensor,
    input_tensor,
    tokenizer,
    labels_list,
):

    print("Decoding augmented ids ...")
    entity_aug_text = []
    o_aug_text = []
    total_aug_text = []

    for entity_aug_ids, o_aug_ids, total_aug_ids, input_ids in zip(
        entity_aug_tensor.tolist(),
        o_aug_tensor.tolist(),
        total_aug_tensor.tolist(),
        input_tensor.tolist(),
    ):
        input_subs = tokenizer.convert_ids_to_tokens(
            input_ids, skip_special_tokens=True
        )
        entity_aug_subs = tokenizer.convert_ids_to_tokens(
            entity_aug_ids, skip_special_tokens=False
        )[
            1 : len(input_subs) + 1
        ]  # Cater for cases when last token is predicted as EOS and thus wrongly removed
        o_aug_subs = tokenizer.convert_ids_to_tokens(
            o_aug_ids, skip_special_tokens=False
        )[1 : len(input_subs) + 1]
        total_aug_subs = tokenizer.convert_ids_to_tokens(
            total_aug_ids, skip_special_tokens=False
        )[1 : len(input_subs) + 1]

        assert len(entity_aug_subs) == len(
            input_subs
        ), f"input {input_subs} \n {input_ids}\n entity_aug {entity_aug_subs}\n{entity_aug_ids}"
        entity_word, o_word, total_word = "", "", ""
        entity_aug_sent, o_aug_sent, total_aug_sent = [], [], []

        special_masks = [f"<{l}>" for l in labels_list]
        for entity_aug_sub, o_aug_sub, total_aug_sub, input_sub in zip(
            entity_aug_subs, o_aug_subs, total_aug_subs, input_subs
        ):
            if input_sub[0] == "▁" or input_sub in special_masks:
                if entity_word != "":
                    entity_aug_sent.append(entity_word)
                entity_word = entity_aug_sub
                if o_word != "":
                    o_aug_sent.append(o_word)
                o_word = o_aug_sub
                if total_word != "":
                    total_aug_sent.append(total_word)
                total_word = total_aug_sub
            else:
                entity_word += entity_aug_sub
                o_word += o_aug_sub
                total_word += total_aug_sub
        # Cater for last word in the sentence
        entity_aug_sent.append(entity_word)
        o_aug_sent.append(o_word)
        total_aug_sent.append(total_word)

        entity_aug_text.append(entity_aug_sent)
        o_aug_text.append(o_aug_sent)
        total_aug_text.append(total_aug_sent)

    return entity_aug_text, o_aug_text, total_aug_text


def inference(args: MELMArguments) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on ", device)

    SEED = args.seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    token_classification_task = NER()
    labels = token_classification_task.get_labels(args.label_path)
    label_map = {"PAD": 0}
    label_map_additional: Dict[str, int] = {
        label: i + 1 for i, label in enumerate(labels)
    }
    label_map.update(label_map_additional)

    entity_model = AutoModelForMaskedLM.from_pretrained(
        args.base_model_name_or_path, return_dict=True
    ).to(device)
    o_model = AutoModelForMaskedLM.from_pretrained(
        args.base_model_name_or_path, return_dict=True
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_name_or_path, do_lower_case=False
    )

    # Add entity labels as special tokens
    # tokenizer.add_tokens(['<En>', '<De>', '<Es>', '<Nl>'], special_tokens=True)
    tokenizer.add_tokens(
        [f"<{label}>" for label in labels], special_tokens=False
    )  # False so that they are not removed during decoding
    entity_model.resize_token_embeddings(len(tokenizer))
    o_model.resize_token_embeddings(len(tokenizer))

    entity_model.load_state_dict(torch.load(os.path.join(args.output_path, "ckpt.pt")))

    dataset = Data(
        tokenizer=tokenizer,
        b_size=args.batch_size,
        label_map=label_map,
        data_path=args.data_path,
        file_name="train.txt",
        mu_ratio=args.mu_ratio,
        sigma=args.sigma,
        mask_rate=args.o_mask_rate,
        labels=labels,
    ).dataset

    dataloader = DataLoader(dataset, batch_size=args.batch_size)

    entity_aug_tensor, o_aug_tensor, total_aug_tensor, input_tensor = aug(
        entity_model, o_model, dataloader, args.k, args.sub_idx
    )
    entity_aug_text, o_aug_text, total_aug_text = decode(
        entity_aug_tensor,
        o_aug_tensor,
        total_aug_tensor,
        input_tensor,
        tokenizer,
        labels,
    )

    num_skip_lines = 0

    for aug_text, ext in zip([entity_aug_text], ["entity"]):
        with open(os.path.join(args.output_path, ".tmp"), "w") as filehandle:  # aug.tmp
            for sent in aug_text:
                for word in sent:
                    filehandle.write("%s\n" % word)  # .lstrip('▁'))
                filehandle.write("\n")

        with open(os.path.join(args.output_path, ".tmp"), "r") as filehandle, open(
            os.path.join(args.output_path, "." + ext), "w+"
        ) as outfile, open(os.path.join(args.data_path, "train.txt"), "r") as infile:

            tmplines = filehandle.readlines()
            inlines = infile.readlines()
            in_idx = 0
            for tmp_idx in range(len(tmplines)):
                special_masks = [f"<{l}>" for l in labels]
                if (
                    tmplines[tmp_idx].rstrip("\n") in special_masks
                ):  # remove special masks
                    continue
                if tmplines[tmp_idx] != "\n":
                    print(inlines[in_idx])
                    # assert inlines[in_idx] != '\n'
                    if inlines[in_idx] == "\n":
                        continue
                    outline = (
                        tmplines[tmp_idx].rstrip("\n").split()[0][1:]
                        + "\t"
                        + inlines[in_idx].rstrip("\n").split()[-1]
                    )
                    outfile.write(outline + "\n")
                else:
                    while inlines[in_idx] != "\n":
                        in_idx += 1
                        num_skip_lines += 1
                    outfile.write("\n")
                in_idx += 1


def train_masked_entity_language_modeling(args: MELMArguments) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    token_classification_task = NER()
    labels = token_classification_task.get_labels(args.label_path)
    label_map = {"PAD": 0}
    label_map_additional: Dict[str, int] = {
        label: i + 1 for i, label in enumerate(labels)
    }
    label_map.update(label_map_additional)

    model = AutoModelForMaskedLM.from_pretrained(
        args.base_model_name_or_path, return_dict=True
    ).to(device)
    tokenizer: Union[
        PreTrainedTokenizer, PreTrainedTokenizerFast
    ] = AutoTokenizer.from_pretrained(args.base_model_name_or_path, do_lower_case=False)

    tokenizer.add_tokens([f"<{label}>" for label in labels])
    model.resize_token_embeddings(len(tokenizer))

    with torch.no_grad():
        if hasattr(model, "roberta"):
            if "conll2003" in args.data_path:
                # initialize label token embedding by using token embedding with almost the same meaning as the label
                model.roberta.embeddings.word_embeddings.weight[
                    -1, :
                ] += model.roberta.embeddings.word_embeddings.weight[1810, :].clone()
                model.roberta.embeddings.word_embeddings.weight[
                    -2, :
                ] += model.roberta.embeddings.word_embeddings.weight[3445, :].clone()
                model.roberta.embeddings.word_embeddings.weight[
                    -3, :
                ] += model.roberta.embeddings.word_embeddings.weight[3445, :].clone()
                model.roberta.embeddings.word_embeddings.weight[
                    -4, :
                ] += model.roberta.embeddings.word_embeddings.weight[53702, :].clone()
                model.roberta.embeddings.word_embeddings.weight[
                    -5, :
                ] += model.roberta.embeddings.word_embeddings.weight[53702, :].clone()
                model.roberta.embeddings.word_embeddings.weight[
                    -6, :
                ] += model.roberta.embeddings.word_embeddings.weight[27060, :].clone()
                model.roberta.embeddings.word_embeddings.weight[
                    -7, :
                ] += model.roberta.embeddings.word_embeddings.weight[27060, :].clone()
                model.roberta.embeddings.word_embeddings.weight[
                    -8, :
                ] += model.roberta.embeddings.word_embeddings.weight[31913, :].clone()
                model.roberta.embeddings.word_embeddings.weight[
                    -9, :
                ] += model.roberta.embeddings.word_embeddings.weight[31913, :].clone()
            elif "multiconer" in args.data_path:
                model.roberta.embeddings.word_embeddings.weight[
                    -1, :
                ] += model.roberta.embeddings.word_embeddings.weight[1810, :].clone()
                model.roberta.embeddings.word_embeddings.weight[
                    -2, :
                ] += model.roberta.embeddings.word_embeddings.weight[12996, :].clone()
                model.roberta.embeddings.word_embeddings.weight[
                    -3, :
                ] += model.roberta.embeddings.word_embeddings.weight[12996, :].clone()
                model.roberta.embeddings.word_embeddings.weight[
                    -4, :
                ] += model.roberta.embeddings.word_embeddings.weight[3445, :].clone()
                model.roberta.embeddings.word_embeddings.weight[
                    -5, :
                ] += model.roberta.embeddings.word_embeddings.weight[3445, :].clone()
                model.roberta.embeddings.word_embeddings.weight[
                    -6, :
                ] += model.roberta.embeddings.word_embeddings.weight[31913, :].clone()
                model.roberta.embeddings.word_embeddings.weight[
                    -7, :
                ] += model.roberta.embeddings.word_embeddings.weight[31913, :].clone()
                model.roberta.embeddings.word_embeddings.weight[
                    -8, :
                ] += model.roberta.embeddings.word_embeddings.weight[94407, :].clone()
                model.roberta.embeddings.word_embeddings.weight[
                    -9, :
                ] += model.roberta.embeddings.word_embeddings.weight[94407, :].clone()
                model.roberta.embeddings.word_embeddings.weight[-10, :] += (
                    model.roberta.embeddings.word_embeddings.weight[[85403, 4488], :]
                    .clone()
                    .mean(0)
                )
                model.roberta.embeddings.word_embeddings.weight[-11, :] += (
                    model.roberta.embeddings.word_embeddings.weight[[85403, 4488], :]
                    .clone()
                    .mean(0)
                )
                model.roberta.embeddings.word_embeddings.weight[
                    -12, :
                ] += model.roberta.embeddings.word_embeddings.weight[216487, :].clone()
                model.roberta.embeddings.word_embeddings.weight[
                    -13, :
                ] += model.roberta.embeddings.word_embeddings.weight[216487, :].clone()

        elif hasattr(model, "deberta"):
            if "ontonotes" in args.data_path:
                # O
                model.deberta.embeddings.word_embeddings.weight[
                    -1, :
                ] += model.deberta.embeddings.word_embeddings.weight[1351, :].clone()

                # WORK_OF_ART
                model.deberta.embeddings.word_embeddings.weight[-2, :] += (
                    model.deberta.embeddings.word_embeddings.weight[
                        [2405, 305, 3685], :
                    ]
                    .clone()
                    .mean(0)
                )
                model.deberta.embeddings.word_embeddings.weight[-3, :] += (
                    model.deberta.embeddings.word_embeddings.weight[
                        [2405, 305, 3685], :
                    ]
                    .clone()
                    .mean(0)
                )

                # TIME
                model.deberta.embeddings.word_embeddings.weight[
                    -4, :
                ] += model.deberta.embeddings.word_embeddings.weight[1460, :].clone()
                model.deberta.embeddings.word_embeddings.weight[
                    -5, :
                ] += model.deberta.embeddings.word_embeddings.weight[1460, :].clone()

                # QUANTITY
                model.deberta.embeddings.word_embeddings.weight[-6, :] += (
                    model.deberta.embeddings.word_embeddings.weight[[260, 28401], :]
                    .clone()
                    .mean(0)
                )
                model.deberta.embeddings.word_embeddings.weight[-7, :] += (
                    model.deberta.embeddings.word_embeddings.weight[[260, 28401], :]
                    .clone()
                    .mean(0)
                )

                # PRODUCT
                model.deberta.embeddings.word_embeddings.weight[
                    -8, :
                ] += model.deberta.embeddings.word_embeddings.weight[5690, :].clone()
                model.deberta.embeddings.word_embeddings.weight[
                    -9, :
                ] += model.deberta.embeddings.word_embeddings.weight[5690, :].clone()

                # PERSON
                model.deberta.embeddings.word_embeddings.weight[
                    -10, :
                ] += model.deberta.embeddings.word_embeddings.weight[2986, :].clone()
                model.deberta.embeddings.word_embeddings.weight[
                    -11, :
                ] += model.deberta.embeddings.word_embeddings.weight[2986, :].clone()

                # PERCENT
                model.deberta.embeddings.word_embeddings.weight[
                    -12, :
                ] += model.deberta.embeddings.word_embeddings.weight[17938, :].clone()
                model.deberta.embeddings.word_embeddings.weight[
                    -13, :
                ] += model.deberta.embeddings.word_embeddings.weight[17938, :].clone()

                # ORG
                model.deberta.embeddings.word_embeddings.weight[
                    -14, :
                ] += model.deberta.embeddings.word_embeddings.weight[29661, :].clone()
                model.deberta.embeddings.word_embeddings.weight[
                    -15, :
                ] += model.deberta.embeddings.word_embeddings.weight[29661, :].clone()

                # ORDINAL
                model.deberta.embeddings.word_embeddings.weight[-16, :] += (
                    model.deberta.embeddings.word_embeddings.weight[[29576, 474], :]
                    .clone()
                    .mean(0)
                )
                model.deberta.embeddings.word_embeddings.weight[-17, :] += (
                    model.deberta.embeddings.word_embeddings.weight[[29576, 474], :]
                    .clone()
                    .mean(0)
                )

                # NORP
                model.deberta.embeddings.word_embeddings.weight[-18, :] += (
                    model.deberta.embeddings.word_embeddings.weight[
                        [260, 263, 144754], :
                    ]
                    .clone()
                    .mean(0)
                )
                model.deberta.embeddings.word_embeddings.weight[-19, :] += (
                    model.deberta.embeddings.word_embeddings.weight[
                        [260, 263, 144754], :
                    ]
                    .clone()
                    .mean(0)
                )

                # MONEY
                model.deberta.embeddings.word_embeddings.weight[
                    -20, :
                ] += model.deberta.embeddings.word_embeddings.weight[8130, :].clone()
                model.deberta.embeddings.word_embeddings.weight[
                    -21, :
                ] += model.deberta.embeddings.word_embeddings.weight[8130, :].clone()

                # LOC
                model.deberta.embeddings.word_embeddings.weight[
                    -22, :
                ] += model.deberta.embeddings.word_embeddings.weight[8939, :].clone()
                model.deberta.embeddings.word_embeddings.weight[
                    -23, :
                ] += model.deberta.embeddings.word_embeddings.weight[8939, :].clone()

                # LAW
                model.deberta.embeddings.word_embeddings.weight[
                    -24, :
                ] += model.deberta.embeddings.word_embeddings.weight[13423, :].clone()
                model.deberta.embeddings.word_embeddings.weight[
                    -25, :
                ] += model.deberta.embeddings.word_embeddings.weight[13423, :].clone()

                # LANGUAGE
                model.deberta.embeddings.word_embeddings.weight[
                    -26, :
                ] += model.deberta.embeddings.word_embeddings.weight[17897, :].clone()
                model.deberta.embeddings.word_embeddings.weight[
                    -27, :
                ] += model.deberta.embeddings.word_embeddings.weight[17897, :].clone()

                # GPE
                model.deberta.embeddings.word_embeddings.weight[-28, :] += (
                    model.deberta.embeddings.word_embeddings.weight[[39128, 28736], :]
                    .clone()
                    .mean(0)
                )
                model.deberta.embeddings.word_embeddings.weight[-29, :] += (
                    model.deberta.embeddings.word_embeddings.weight[[39128, 28736], :]
                    .clone()
                    .mean(0)
                )

                # FAC
                model.deberta.embeddings.word_embeddings.weight[-30, :] += (
                    model.deberta.embeddings.word_embeddings.weight[[23634, 277], :]
                    .clone()
                    .mean(0)
                )
                model.deberta.embeddings.word_embeddings.weight[-31, :] += (
                    model.deberta.embeddings.word_embeddings.weight[[23634, 277], :]
                    .clone()
                    .mean(0)
                )

                # EVENT
                model.deberta.embeddings.word_embeddings.weight[
                    -32, :
                ] += model.deberta.embeddings.word_embeddings.weight[10276, :].clone()
                model.deberta.embeddings.word_embeddings.weight[
                    -33, :
                ] += model.deberta.embeddings.word_embeddings.weight[10276, :].clone()

                # DATE
                model.deberta.embeddings.word_embeddings.weight[
                    -34, :
                ] += model.deberta.embeddings.word_embeddings.weight[5256, :].clone()
                model.deberta.embeddings.word_embeddings.weight[
                    -35, :
                ] += model.deberta.embeddings.word_embeddings.weight[5256, :].clone()

                # CARDINAL
                model.deberta.embeddings.word_embeddings.weight[-36, :] += (
                    model.deberta.embeddings.word_embeddings.weight[[260, 168193], :]
                    .clone()
                    .mean(0)
                )
                model.deberta.embeddings.word_embeddings.weight[-37, :] += (
                    model.deberta.embeddings.word_embeddings.weight[[260, 168193], :]
                    .clone()
                    .mean(0)
                )

            elif "multiconer" in args.data_path:
                model.deberta.embeddings.word_embeddings.weight[
                    -1, :
                ] += model.deberta.embeddings.word_embeddings.weight[1351, :].clone()

                model.deberta.embeddings.word_embeddings.weight[
                    -2, :
                ] += model.deberta.embeddings.word_embeddings.weight[14972, :].clone()
                model.deberta.embeddings.word_embeddings.weight[
                    -3, :
                ] += model.deberta.embeddings.word_embeddings.weight[14972, :].clone()
                model.deberta.embeddings.word_embeddings.weight[
                    -4, :
                ] += model.deberta.embeddings.word_embeddings.weight[2986, :].clone()
                model.deberta.embeddings.word_embeddings.weight[
                    -5, :
                ] += model.deberta.embeddings.word_embeddings.weight[2986, :].clone()
                model.deberta.embeddings.word_embeddings.weight[
                    -6, :
                ] += model.deberta.embeddings.word_embeddings.weight[8939, :].clone()
                model.deberta.embeddings.word_embeddings.weight[
                    -7, :
                ] += model.deberta.embeddings.word_embeddings.weight[8939, :].clone()
                model.deberta.embeddings.word_embeddings.weight[
                    -8, :
                ] += model.deberta.embeddings.word_embeddings.weight[36059, :].clone()
                model.deberta.embeddings.word_embeddings.weight[
                    -9, :
                ] += model.deberta.embeddings.word_embeddings.weight[36059, :].clone()
                model.deberta.embeddings.word_embeddings.weight[-10, :] += (
                    model.deberta.embeddings.word_embeddings.weight[[45050, 2405], :]
                    .clone()
                    .mean(0)
                )
                model.deberta.embeddings.word_embeddings.weight[-11, :] += (
                    model.deberta.embeddings.word_embeddings.weight[[45050, 2405], :]
                    .clone()
                    .mean(0)
                )
                model.deberta.embeddings.word_embeddings.weight[
                    -12, :
                ] += model.deberta.embeddings.word_embeddings.weight[96243, :].clone()
                model.deberta.embeddings.word_embeddings.weight[
                    -13, :
                ] += model.deberta.embeddings.word_embeddings.weight[96243, :].clone()

    train_dataset, valid_dataset = tuple(
        DataForTrainer(
            tokenizer=tokenizer,
            b_size=args.batch_size,
            label_map=label_map,
            file_dir=args.data_path,
            mask_rate=args.mask_rate,
            labels=labels,
        ).datasets
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=RandomSampler(train_dataset)
    )
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size)
    # test_dataloader = DataLoader(test_dataset, batch_size=BSIZE)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_valid_loss = float("inf")
    best_valid_entity_acc = -float("inf")
    best_valid_entity_acc_by_acc = -float("inf")

    for epoch in range(args.n_epochs):
        start_time = time.time()

        train(model, train_dataloader, optimizer, args.clip, args.grad_acc, epoch)
        valid_loss, valid_acc, valid_entity_acc = evaluate(model, valid_dataloader)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        print(
            f"Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s",
            f"Epoch valid loss: {valid_loss:.3f} | ",
            f"Epoch valid acc: {valid_acc * 100:.2f}% | Epoch entity acc: {valid_entity_acc*100:.2f}% ",
        )

        if valid_loss < best_valid_loss:
            print("Saving current epoch to checkpoint...")
            best_valid_loss = valid_loss
            best_valid_epoch = epoch
            best_valid_acc = valid_acc
            best_valid_entity_acc = valid_entity_acc
            torch.save(model.state_dict(), os.path.join(args.output_path, "ckpt.pt"))

    print("Training finished...")
    print(
        f"\n Best valid loss until epoch {epoch} is {best_valid_loss:.3f} at epoch {best_valid_epoch + 1}",
        f"\n valid acc is {best_valid_acc * 100:.2}%, valid entity acc is {best_valid_entity_acc * 100:.2f}%",
    )


def convert_linearized_tokens_to_iob2(
    original_list: list[str],
) -> tuple[list[str], list[str]]:
    tokens = []
    tags = []
    for i in range(0, len(original_list)):
        if i == len(original_list) - 1:
            if original_list[i].startswith("<") and original_list[i].endswith(">"):
                continue
            else:
                tokens.append(original_list[i].replace("▁", ""))
                tags.append("O")

        elif (
            original_list[i - 1].startswith("<")
            and original_list[i - 1].endswith(">")
            and original_list[i + 1]
            and original_list[i + 1].endswith(">")
            and original_list[i - 1][1:-1] == original_list[i + 1][1:-1]
        ):
            tokens.append(original_list[i].replace("▁", ""))
            tags.append(original_list[i - 1][1:-1])

        elif original_list[i].startswith("<") and original_list[i].endswith(">"):
            continue

        else:
            tokens.append(original_list[i].replace("▁", ""))
            tags.append("O")

    return tokens, tags


def check_existence(sublist: list, full_list: list) -> bool:
    for i, l in enumerate(full_list):
        if l == sublist[0] and sublist == full_list[i : i + len(sublist)]:
            return True

    return False


def inspection(raw_list: list[str], tags: list[str]):
    for i in range(len(tags) - 2):
        if tags[i] == tags[i + 1] == tags[i + 2] and tags[i].startswith("B-"):
            label_name = tags[i]

            if not check_existence([f"<{label_name}>" * 2], raw_list):
                tags[i + 1] = "O"

    return tags


def get_augmented_dataset_from_full_process(args: MELMArguments) -> Dataset:

    train_masked_entity_language_modeling(args)
    inference(args)

    tokens_tags_path = os.path.join(args.output_path, ".tmp")

    tokens = []
    total_tokens = []

    with open(tokens_tags_path) as f:
        for line in f:
            if line != "\n":
                tokens.append(line.strip("\n"))
            else:
                total_tokens.append(tokens)
                tokens = []

    final_tokens = []
    final_tags = []
    records = []
    for i, original_list in enumerate(total_tokens):
        tokens, tags = convert_linearized_tokens_to_iob2(original_list)
        try:
            tags = inspection(original_list, tags)
        except IndexError:
            pass
        final_tokens.append(tokens)
        final_tags.append(tags)
        records.append(
            {
                "id": i,
                "tokens": [re.sub(r"<B-.*>", "", token) for token in tokens],
                "ner_tags": tags,
            }
        )

    with open(os.path.join(args.output_path, "augmented_data.json"), "w") as f:
        json.dump(records, f, indent=4, ensure_ascii=False)

    full_dict = {
        "id": list(range(len(final_tokens))),
        "tokens": final_tokens,
        "ner_tags": final_tags,
    }
    return Dataset.from_dict(full_dict)


if __name__ == "__main__":
    args = MELMArguments().parse_args()
    dataset = get_augmented_dataset_from_full_process(args)
