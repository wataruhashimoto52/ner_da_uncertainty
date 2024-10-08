import json
import random

from algorithms.augmentators import IOB2, LabelWiseTokenReplacement, SynonymReplacement, MentionReplacement
from tap import Tap
from transformers import set_seed

set_seed(42)


class Arguments(Tap):
    data_name: str
    algorithm: str
    swap_ratio: float
    augment_size: float

    def configure(self) -> None:
        self.add_argument("--data_name", type=str, required=True)
        self.add_argument("--algorithm", type=str, required=True)
        self.add_argument("--swap_ratio", type=float, required=True)
        self.add_argument("--augment_size", type=float, required=True)

def main(args: Arguments) -> None:
    with open(f"data/ner_data/{args.data_name}/train.json") as f:
        train_data: list[dict] = json.load(f)

    X_base = [d["tokens"] for d in train_data]
    Y_base = [d["ner_tags"] for d in train_data]

    if args.algorithm == "LabelWiseTokenReplacement":
        augmentator = LabelWiseTokenReplacement(
            x=X_base,
            y=Y_base,
            p=args.swap_ratio,
        )
    elif args.algorithm == "MentionReplacement":
        augmentator = MentionReplacement(
            x=X_base,
            y=Y_base,
            scheme=IOB2,
            p=args.swap_ratio,
        )

    elif args.algorithm == "SynonymReplacement":
        augmentator = SynonymReplacement(p=args.swap_ratio)
    else:
        raise NotImplementedError

    records: list[dict] = []

    shuffled_train_data = random.shuffle(train_data, len(train_data))
    cnt = 0
    for i, sample_data in enumerate(shuffled_train_data):
        if cnt >= int(len(train_data) * args.augment_size):
            continue

        x_aug, y_aug = augmentator.augment(
            x=sample_data["tokens"], y=sample_data["ner_tags"]
        )

        if x_aug[0] != sample_data["tokens"]:
            cnt += 1
            if "ontonotes" in args.data_name:
                records.append(
                    {
                        "id": len(train_data) + cnt,
                        "tokens": x_aug[0],
                        "ner_tags": y_aug[0],
                    }
                )
            else:
                records.append(
                    {
                        "domain": len(train_data) + cnt,
                        "tokens": x_aug[0],
                        "ner_tags": y_aug[0],
                    }
                )

    train_data.extend(records)

    with open(f"data/ner_data/{args.algorithm}_{args.data_name}/train.json", "w") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":

    args = Arguments().parse_args()
    main(args)
