from pathlib import Path
import re
from random import sample, seed
import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator


class CRprocessing:
    """Encompasses all of the preprocessing stack for the CRgen project
    Provides access to all of the intermediary stages in variables created along the way at init

    """

    def __init__(
        self,
        string_path,
        n_sentences_per_bundle=400,
        prop_train=0.7,
        prop_test=0.8,
        tokenizer_name="basic_english",
    ):
        """Preprocess the text files found in path directory

        Does:
        Cleaning
        Bundling
        Shuffling and splitting
        Encoding
        Shipping to torch Tensors

        Refer to the documentations of each step for more information

        Arguments:
            string_path {str} -- [the path to the folder containing the crT txt files to process]

        Keyword Arguments:
            n_sentences_per_bundle {number} -- [max number of sentences to put in a bundle] (default: {400})
            prop_train {number} -- [proportion of the data to use for the train set] (default: {.7})
            prop_test {number} -- [proportion of the non-training data to use for the train set (the rest will go in the eval set)] (default: {.8})
            tokenizer_name {str} -- [the name of the torchtext tokenizer] (default: {'basic_english'})
        """
        self.data_dir = Path(string_path)
        self.n_sen_per_bundle = n_sentences_per_bundle
        self.prop_train = prop_train
        self.prop_test = prop_test
        self.tokenizer = get_tokenizer(tokenizer_name)
        self.ep_texts = []
        self.text_data = []
        self.CRcleaning()
        self.CRbundling()
        self.CRtrain_test_eval()
        self.CRcreate_vocab(self.train_set)
        self.CRtext_to_torch()

    def CRcleaning(self):
        """Reads the data from the provided folder and ships it to a list of string
        Cleans line breaks and double line breaks

        Provides attribute ep_texts for raw episode string data
        """
        for text_file in self.data_dir.iterdir():
            content = text_file.read_text(encoding="utf-8")
            cleaned = content.replace("\n\n", " ").replace("\n", " ")
            self.ep_texts.append(cleaned)

    def CRbundling(self):
        """Splits episodes without breaking sentences, and then bundles them into lists of string

        Provides attribute text_data for list of bundles
        """
        for ep in self.ep_texts:
            i = 0
            sentences = re.split("([.!?])", ep)  # we don't want to lose the delimiters
            n = self.n_sen_per_bundle * 2
            while i < len(sentences):
                self.text_data.append("".join(sentences[i : i + n]))
                i += n

    def shuffle_split(self, l, prop=0.5):
        """
        Shuffles the data in l before splitting it in two lists according to prop

        Args:
            prop {float} -- the proportion of data to be stored in the first list returned
        """
        seed(1122)  # reproductability
        n = len(l)
        shuffled = sample(l, n)  # create a shuffled version of l (not in-place)
        size_of_prop = int(n * prop)
        return shuffled[:size_of_prop], shuffled[size_of_prop:]

    def CRtrain_test_eval(self):
        """Shuffles and splits original data into attributes train_set, test_set, and eval_set

        cf shuffle_split
        """

        self.train_set, non_train_set = self.shuffle_split(
            self.text_data, self.prop_train
        )
        self.test_set, self.eval_set = self.shuffle_split(non_train_set, self.prop_test)

    def CRcreate_vocab(self, set_for_vocab):
        """builds this preprocessing instance's vocabulary

        Provides attribute vocab for translation in each direction (cf torchtext vocab object documentation)

        Arguments:
            set_for_vocab {iterable} -- [the set of text to use to build the vocabulary]
        """
        print("Building vocabulary...")
        self.vocab = build_vocab_from_iterator(map(self.tokenizer, set_for_vocab))

    def encoding_to_torch(self, text_iter):
        """
        Tokenizes, Encodes, and ships to a torch.tensor the data from an iterable set of text

        Note: the vocabulary needed for the encoding should have been built before calling this function
        """
        data = [
            torch.tensor(
                [self.vocab[token] for token in self.tokenizer(item)], dtype=torch.long
            )
            for item in text_iter
        ]
        # we get rid of any empty encodings before returning
        return (
            torch.cat(tuple(filter(lambda t: t.numel() > 0, data))),
            len(self.vocab.stoi),
        )

    def CRtext_to_torch(self):
        """encodes and ships to tensors the three sets

        Provides train_data_encoded, test_data_encoded, eval_data_encoded to use for training or batching
        """
        self.train_data_encoded = self.encoding_to_torch(self.train_set)
        self.eval_data_encoded = self.encoding_to_torch(self.eval_set)
        self.test_data_encoded = self.encoding_to_torch(self.test_set)


# cleaning
# encode text (AE?)
# bundle text in chunks
# Save in a dir
# Dataset
# DataLoader ? (training code)
