import os
from pathlib import Path
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from itertools import chain


# Dataset
# DataLoader ? (training code)*


def custom_tokenizer(string, wrong_file="Words_to_stitch.txt"):
    """Improves upon the basic english tokenizer by
    stitching together the We'll and the It's etc"""

    with open(wrong_file, "r") as f:
        wrongly_tokenized = f.read().split("\n")

    norm_split = get_tokenizer("basic_english")(string)  # normalize and split on spaces
    spread_words = flatten_list(
        [word.split("--") for word in norm_split]
    )  # split further on --
    tokens = []
    for i, word in enumerate(spread_words):
        if word == "'":
            potential_word = spread_words[i - 1] + word + spread_words[i + 1]
            if potential_word in wrongly_tokenized:
                tokens.append(potential_word)
        else:
            tokens.append(word)

    return tokens


TOKENIZER = get_tokenizer(custom_tokenizer)


def clean_episodes(data_dir):
    all_episode_texts = []
    for text_file in Path(data_dir).iterdir():
        content = text_file.read_text(encoding="utf-8")
        cleaned = content.replace("\n\n", " ").replace("\n", " ")
        all_episode_texts.append(cleaned)

    return all_episode_texts


def flatten_list(outer_list):
    return list(chain(*outer_list))


def split_into_bundles(episodes_list, n_sentences_per_bundle):
    """Splits the input into bundles without breaking sentences"""


def full_preprocessing_cycle(
    data_dir, n_sentences_per_bundle, output_dir="../data/ready"
):
    # cleaning
    # encode text (AE?)
    # bundle text in chunks
    # Save in a dir

    os.makedirs(output_dir, exist_ok=True)
    episodes_list = clean_episodes(data_dir)
    tokenized_eps = [TOKENIZER(episode) for episode in episodes_list]
    all_tokens = flatten_list(tokenized_eps)
    vocab = build_vocab_from_iterator(
        iterator=iter(all_tokens), min_freq=1, max_tokens=None, specials="unk"
    )
    bundled = split_into_bundles(tokenized_eps)

    print(f"Saved preprocessed data to {output_dir}")


if __name__ == "__main__":
    data_dir = "../data/original/"
    full_preprocessing_cycle(data_dir)
