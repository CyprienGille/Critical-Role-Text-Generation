from pathlib import Path
import re


def split_by_speaker(content: str):
    # this regular expression finds a string of more than one capitalized letter, followed by a ':'
    all_speakers = list(re.finditer(r"([A-Z])+:", content))
    tirades = []
    for match_i in range(len(all_speakers) - 1):
        tirades.append(
            content[
                all_speakers[match_i].span()[0] : all_speakers[match_i + 1].span()[0]
            ]
        )

    tirades.append(content[all_speakers[-1].span()[0] :])
    return tirades


def get_processed_strings(data_dir: str, min_len=None):
    processed = []
    for text_file in Path(data_dir).iterdir():
        # read episode file
        content = text_file.read_text(encoding="utf-8")
        # remove line breaks
        content = content.replace("\n\n", " ").replace("\n", " ")
        # split by person speaking
        tirades = split_by_speaker(content)

        if min_len is not None:
            # if we want to filter out sentences that are less than min_len words long
            tirades = [t for t in tirades if len(t.split(" ")) > min_len]

        # add all tirades to the processed strings list
        processed.extend(tirades)

    return processed
