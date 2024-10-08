"""
ref.

https://github.com/boschresearch/data-augmentation-coling2020
https://github.com/kajyuuen/daaja
https://github.com/Hironsan/neraug
"""

import abc
import random
import string
from collections import Counter, defaultdict
from itertools import chain
from typing import Callable, Dict, List, Type

import nltk
import numpy as np
from nltk.corpus import stopwords, wordnet
from seqeval import scheme as s


IOB2 = s.IOB2
BILOU = s.BILOU
IOBES = s.IOBES
Token = s.Token
Entities = s.Entities


def create_tagger(scheme: Type[Token]):
    if scheme == IOB2:
        return IOB2Tagger()
    elif scheme == IOBES:
        return StartInsideEndTagger()
    elif scheme == BILOU:
        return StartInsideEndTagger("U", "L")
    else:
        raise ValueError(f"The scheme value is invalid: {scheme}")


class BaseTagger(abc.ABC):
    @abc.abstractmethod
    def tag(self, words: List[str], label: str):
        assert len(words) > 0


class IOB2Tagger(BaseTagger):
    def tag(self, words: List[str], label: str):
        super().tag(words, label)
        return [f"B-{label}"] + [f"I-{label}"] * (len(words) - 1)


class StartInsideEndTagger(BaseTagger):
    def __init__(self, single="S", end="E"):
        self.single = single
        self.end = end

    def tag(self, words: List[str], label: str):
        super().tag(words, label)
        if len(words) == 1:
            return [f"{self.single}-{label}"]
        else:
            return [f"B-{label}"] + [f"I-{label}"] * (len(words) - 2) + [f"{self.end}-{label}"]


class BaseReplacement(abc.ABC):
    @abc.abstractmethod
    def augment(self, x: List[str], y: List[str], n=1):
        raise NotImplementedError()


class DictionaryReplacement(BaseReplacement):
    def __init__(
        self,
        ne_dic: Dict[str, str],
        tokenize: Callable[[str], List[str]],
        scheme: Type[Token],
    ):
        self.dic = defaultdict(list)
        tagger = create_tagger(scheme)
        for entity, label in ne_dic.items():
            words = tokenize(entity)
            self.dic[label].append({"words": words, "tags": tagger.tag(words, label)})
        self.scheme = scheme

    def augment(self, x: List[str], y: List[str], n=1):
        xs = []
        ys = []
        entities = Entities([y], self.scheme)
        for i in range(n):
            x_ = []
            y_ = []
            start = 0
            for entity in chain(*entities.entities):
                if not self.dic[entity.tag]:
                    continue
                data = random.choice(self.dic[entity.tag])
                x_.extend(x[start : entity.start])
                x_.extend(data["words"])
                y_.extend(y[start : entity.start])
                y_.extend(data["tags"])
                start = entity.end
            x_.extend(x[start:])
            y_.extend(y[start:])
            xs.append(x_)
            ys.append(y_)
        return xs, ys


class SynonymReplacement(BaseReplacement):
    def __init__(
        self,
        p: float = 0.3,
    ) -> None:
        self.p = p
        nltk.download("stopwords")
        nltk.download("wordnet")
        self._load_stop_words()

    def _load_stop_words(self) -> None:
        self.stop_words: set[str] = set(stopwords.words("english"))

    def get_synonyms(self, word: str) -> list[str]:
        synonyms: list[str] = [
            name for syn in wordnet.synsets(word) for name in syn.lemma_names()
        ]

        synonyms_set = set(synonyms)
        if word in synonyms_set:
            synonyms_set.remove(word)
        return list(synonyms_set)

    def is_stopwords(self, word: str) -> bool:
        return word in self.stop_words

    def augment(
        self,
        x: list[str],
        y: list[str],
        n: int = 1,
    ) -> tuple[list[list[str]], list[list[str]]]:
        masks = np.random.binomial(1, self.p, len(x))
        generated_tokens = []
        for mask, token, label in zip(masks, x, y):
            if mask == 0 or self.is_stopwords(token):
                generated_token = token
            else:
                synonyms_set = set(self.get_synonyms(token))
                if token in synonyms_set:
                    synonyms_set.remove(token)
                if len(synonyms_set) == 0:
                    generated_token = token
                else:
                    synonym = random.choice(list(synonyms_set))
                    generated_token = synonym

            generated_tokens.append(generated_token)

        return [generated_tokens], [y]


class MentionReplacement(BaseReplacement):
    def __init__(
        self,
        x: List[List[str]],
        y: List[List[str]],
        scheme: Type[Token],
        p: float = 0.8,
    ):
        entities = Entities(y, scheme)
        dic = defaultdict(list)
        for tag in entities.unique_tags:
            for entity in entities.filter(tag):
                if random.random() > p:
                    continue
                words = x[entity.sent_id][entity.start : entity.end]
                tags = y[entity.sent_id][entity.start : entity.end]
                dic[tag].append({"words": words, "tags": tags})
        self.replacement = DictionaryReplacement({}, str.split, scheme)
        self.replacement.dic = dic

    def augment(self, x: List[str], y: List[str], n=1):
        return self.replacement.augment(x, y, n)
    
    
class LabelWiseTokenReplacement(BaseReplacement):
    def __init__(self, x: List[List[str]], y: List[List[str]], p=0.8):
        self.p = p
        self.distribution = defaultdict(Counter)
        for words, tags in zip(x, y):
            for word, tag in zip(words, tags):
                self.distribution[tag][word] += 1

    def augment(self, x: List[str], y: List[str], n=1):
        xs = []
        ys = []
        for i in range(n):
            x_ = []
            for word, tag in zip(x, y):
                if random.random() <= self.p:
                    counter = self.distribution[tag]
                    words = list(counter.keys())
                    weights = list(counter.values())
                    word = random.choices(words, weights=weights, k=1)[0]
                x_.append(word)
            xs.append(x_)
            ys.append(y)
        return xs, ys


class RandomMaskingReplacement(BaseReplacement):
    def __init__(
        self,
        p: float = 0.3,
    ) -> None:
        self.p = p
        nltk.download("stopwords")
        self._load_stop_words()

    def _load_stop_words(self) -> None:
        self.stop_words: set[str] = set(stopwords.words("english"))

    def is_stopwords(self, word: str) -> bool:
        return word in self.stop_words

    def augment(
        self,
        x: list[str],
        y: list[str],
        n: int = 1,
    ) -> tuple[list[list[str]], list[list[str]]]:
        generated_tokens = []
        for token, label in zip(x, y):
            if label == "O":
                aug_token = token
            else:
                mask = np.random.binomial(1, self.p)
                if mask == 0:  # no masking
                    aug_token = token
                else:  # masking
                    if self.is_stopwords(token):
                        aug_token = token
                    else:
                        aug_token = ""
                        for char in list(token):
                            if "a" <= char <= "z":
                                aug_token += random.choice(string.ascii_lowercase)
                            elif "A" <= char <= "Z":
                                aug_token += random.choice(string.ascii_uppercase)
                            else:
                                aug_token += char

            generated_tokens.append(aug_token)

        return [generated_tokens], [y]
