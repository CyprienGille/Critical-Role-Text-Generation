from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import gradio as gr

tokenizer = GPT2Tokenizer.from_pretrained("logs/checkpoint-56500")
model = GPT2LMHeadModel.from_pretrained("logs/checkpoint-56500")


def generator(input_string):
    input_string = "<|startoftext|>" + " " * (input_string != "") + input_string
    prompt = torch.tensor(tokenizer.encode(input_string)).unsqueeze(0)

    generated = model.generate(
        prompt,
        do_sample=True,
        top_k=50,
        max_length=1024,
        top_p=0.95,
        num_return_sequences=5,
    )
    out = ""
    for tirade in generated:
        out += tokenizer.decode(tirade, skip_special_tokens=True) + "\n\n"
    return out


desc = "> 'Artificial Intelligence, Eh? Sounds fancy - but it'll never replace geniuses such as myself.' - *Huron Stahlmast, Exiled Hupperdook Engineer*\n\nThis generator allows you to generate your own transcripts from an imaginary episode of Critical Role! Input the start of a tirade (or nothing!), and let the magic of machine learning do the rest!\n\nFor the curious among you, this uses a fine-tuned version of GPT2."

demo = gr.Interface(
    fn=generator,
    inputs="textbox",
    outputs="textbox",
    description=desc,
    title="Critical Role Text Generator",
)
demo.launch()
