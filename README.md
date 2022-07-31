# Text Generation based on Critical Role scripts

This language modeling project teaches a Transformer network to continue sentences, by training it on a collection of transcripts of the live show Critical Role.

## The Dataset
Our dataset is comprised of all of the episode transcripts realised by the team over at CRTranscript. It contains *180 files*, for a total of over *7 million words*, with a vocabulary of *52123 different tokens*.

After pre-processing, i.e. running the `preprocess.py` script, this dataset is bundled into *58171 133-words long sequences*. 

## The Model
The Model is a Transformer, full with both its Encoder and Decoder sections. It also contains a positional encoding module, as well as an Embedder. All of those can be found in the `models.py` file.

## How to use
- Clone the repository locally
- Make sure you have PyTorch installed (as well as numpy and pandas)
- Run `preprocess.py` to get all of the bundles in the newly created `data/ready/` directory.
- Adjust the parameters at the start of `Training.py` and run it to train a model and save it in the `data/` directory.
- Change the input sequence and the number of words to be generated in `main.py` and run it, and voilà! You just generated brand new words from an imaginary Critical Role Episode.