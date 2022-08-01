![image](https://user-images.githubusercontent.com/83751996/182046417-b53b556f-4ae0-44fc-9c7d-3ac72d39a299.png)

# Text Generation based on Critical Role scripts 📺

This language modeling project teaches a [Transformer](https://en.wikipedia.org/wiki/Transformer_(machine_learning_model)) network to continue sentences, by training it on a collection of transcripts of the live show [Critical Role](https://critrole.com/).

## The Dataset
Our dataset is comprised of all of the episode transcripts realised by the team over at [CRTranscript](https://crtranscript.tumblr.com/transcripts). It contains *180 files*, for a total of over *7 million words*, with a vocabulary of *52123 different tokens*.

After pre-processing, i.e. running the `preprocess.py` script, this dataset is bundled into *58171 133-words long sequences*. 

## The Model
The Model uses a [PyTorch Transformer](https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html), full with both its Encoder and Decoder sections. It also contains a positional encoding module, as well as an Embedder. All of those can be found in the `models.py` file.

## How to use
- Clone the repository locally
- Make sure you have [PyTorch](https://pytorch.org/get-started/locally/) installed (as well as [numpy](https://numpy.org/) and [pandas](https://pandas.pydata.org/))
- Run `preprocess.py` to get all of the bundles in the newly created `data/ready/` directory.
- Adjust the parameters at the start of `Training.py` and run it to train a model and save it in the `data/` directory.
- Change the input sequence and the number of words to be generated in `main.py` and run it, and voilà! You just generated brand new words from an imaginary Critical Role Episode.
