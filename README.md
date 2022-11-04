![image](https://user-images.githubusercontent.com/83751996/182046417-b53b556f-4ae0-44fc-9c7d-3ac72d39a299.png)

# Text Generation based on Critical Role scripts 📺

This language modeling project teaches a [Transformer](https://en.wikipedia.org/wiki/Transformer_(machine_learning_model)) network to continue sentences, by training it on a collection of transcripts of the live show [Critical Role](https://critrole.com/).

## The Dataset
Our dataset is comprised of all of the episode transcripts realised by the team over at [CRTranscript](https://crtranscript.tumblr.com/transcripts). It contains *180 files*, for a total of over *7 million words*, with a vocabulary of *52123 different tokens*.
 

## The Model
This project uses a [PyTorch Transformer](https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html) : it fine-tunes an instance of GPT2 using the [HuggingFace API](https://huggingface.co/docs/transformers/index). 


## The App
A [Gradio App](https://www.gradio.app/docs/) showcasing the capabilities of the trained model is available at : https://huggingface.co/spaces/Callimethee/Imagine-CR . Have fun with it!