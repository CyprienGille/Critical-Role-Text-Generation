from pathlib import Path
import re


def split_by_speaker(content: str):
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


def get_processed_strings(data_dir: str):
    processed = []
    for text_file in Path(data_dir).iterdir():
        # read episode file
        content = text_file.read_text(encoding="utf-8")
        # remove line breaks
        content = content.replace("\n\n", " ").replace("\n", " ")

        tirades = split_by_speaker(content)
        # add all tirades to the processed strings list
        processed.extend(tirades)

    return processed
