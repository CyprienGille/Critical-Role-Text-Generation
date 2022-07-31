#%%
import os
from pathlib import Path
import re
import numpy as np
from torch import save
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


def split_into_bundles(all_tokens, wpb):
    """Splits the input into bundles
    Note: this cuts off some words at the end to avoid an incomplete bundle"""
    return [all_tokens[i : i + wpb] for i in range(len(all_tokens) // wpb)]


def save_bundle(bundle, index, path):
    np.save(f"{path}bundle_{index}.npy", np.array(bundle))


def full_preprocessing_cycle(
    data_dir, words_per_bundle=130, output_dir="../data/ready/"
):
    # cleaning
    # encode text
    # bundle text in chunks
    # Save in a dir

    os.makedirs(output_dir, exist_ok=True)

    print("Cleaning episodes...")
    episodes_list = clean_episodes(data_dir)

    print("Tokenizing episodes...")
    tokenized_e = [TOKENIZER(ep) for ep in episodes_list]
    all_tokens = flatten_list(tokenized_e)

    print("Bundling words...")
    bundled_data = split_into_bundles(all_tokens, wpb=words_per_bundle)

    print("Building vocab...")
    # build vocab from all words
    vocab = build_vocab_from_iterator([all_tokens])
    save(vocab, "../data/vocab.pth")
    print(f"Saved vocab with size : {len(vocab)}")

    print("Encoding bundles...")
    encoded_b = [[vocab[word] for word in bundle] for bundle in bundled_data]

    print("Saving bundles...")
    for i, bundle in enumerate(encoded_b):
        save_bundle(bundle, i, path=output_dir)

    print(f"Saved preprocessed data to {output_dir}")


#%%
if __name__ == "__main__":
    data_dir = "../data/original/"
    full_preprocessing_cycle(data_dir)

# %%
