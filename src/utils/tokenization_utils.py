from typing import Any, Union

from transformers import BatchEncoding, PreTrainedTokenizer


def tokenize_and_align_labels_from_json(
    examples: dict[str, Any],
    tokenizer: PreTrainedTokenizer,
    padding: Union[bool, str],
    label2id: dict[str, int],
):

    examples_labels = [[label2id[tag] for tag in tags] for tags in examples["ner_tags"]]
    tokenized_inputs: BatchEncoding = tokenizer(
        examples["tokens"],
        padding=padding,
        truncation=True,
        is_split_into_words=True,
    )
    labels = []
    valid_masks = []
    for i, label in enumerate(examples_labels):
        word_ids = [None]
        valid_mask = [1]
        for j, word in enumerate(examples["tokens"][i]):
            token = tokenizer.encode(word, add_special_tokens=False)
            word_ids += [j] * len(token)
            for k in range(len(token)):
                valid_mask.append(1 if k == 0 else 0)

        word_ids += [None]
        valid_mask += [1]

        if padding == "max_length":
            valid_mask = valid_mask[0 : tokenizer.model_max_length]
            valid_mask += [
                0 for _ in range(tokenizer.model_max_length - len(valid_mask))
            ]
        valid_masks.append(valid_mask)

        # word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx

        if padding == "max_length":
            label_ids += [
                -100 for _ in range(tokenizer.model_max_length - len(label_ids))
            ]

        assert len(label_ids) == len(valid_mask)
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    tokenized_inputs["valid_masks"] = valid_masks

    return tokenized_inputs
