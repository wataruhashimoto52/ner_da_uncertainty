# Are Data Augmentation Methods in Named Entity Recognition Applicable for Uncertainty Estimation?


# Usage

To reproduce our result, you need to run following processes.

## Setup

We use Singularity environment.

```bash
module load singularity
singularity build --fakeroot research.sif research.def
```


## Prepare datasets

Prepare and convert datasets to json from the following formats, 


### OntoNotes 5.0

https://catalog.ldc.upenn.edu/LDC2013T19 

```
[
    {
        "id": 1,
        "tokens": [
            "I",
            "am",
            "Richard",
            "Feynman",
        ],
        "tags": [
            "O",
            "O",
            "B-PER",
            "I-PER",
        ]
    },
    ...
]
```


### MultiCoNER

https://multiconer.github.io/ 

```
[
    {
        "domain": "train",
        "tokens": [
            "I",
            "am",
            "Richard",
            "Feynman",
        ],
        "tags": [
            "O",
            "O",
            "B-PER",
            "I-PER",
        ]
    },
    ...
]
```


## Data augmentation

Run data augmentation methods, and then merge with the original training data.


### Label-wise token replacement, Mention replacement, Synonym replacement (Dai and Adel, 2020)

paper: https://aclanthology.org/2020.coling-main.343/ 


```bash
python src/augment.py \
  --data_name ontonotes5_bn \
  --algorithm LabelWiseTokenReplacement \
  --swap_ratio 0.3 \
  --augment_size 0.5
```


### Masked Entity Language Modeling (Zho et al, 2022)

paper: https://aclanthology.org/2022.acl-long.160/

```bash
sbatch batch_run_melm.sh
```


## Train & Evaluate

```bash
sbatch batch_run_custom_ner.sh
```
