from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import gradio as gr

tokenizer = GPT2Tokenizer.from_pretrained("logs/checkpoint-13500")
model = GPT2LMHeadModel.from_pretrained("logs/checkpoint-13500")


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


demo = gr.Interface(fn=generator, inputs="textbox", outputs="textbox")
demo.launch()
