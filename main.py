#%%
# Module Imports
from math import trunc
from preprocess import get_processed_strings
from transformers import (
    GPT2Tokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
import torch
from torch.utils.data import Dataset, random_split
from tqdm import tqdm

#%%
# Script constants
VERBOSE = False


#%%
# Define Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(
    "gpt2", bos_token="<|startoftext|>", eos_token="<|endoftext|>", pad_token="<|pad|>"
)

if VERBOSE:
    print(tokenizer.encode("MATT: Hello everyone"))

#%%
# Define custom Dataset
class CRDataset(Dataset):
    def __init__(self, str_list, tokenizer, max_len):
        super().__init__()

        if max_len > 1024:
            max_len = (
                1024  # max len accepted by the tokenizer, will truncate the tirades.
            )

        self.input_ids = []
        self.attn_masks = []

        for txt in tqdm(str_list):
            encodings_dict = tokenizer(
                "<|startoftext|>" + txt + "<|endoftext|>",
                truncation=True,
                max_length=max_len,
                padding="max_length",
            )

            self.input_ids.append(torch.tensor(encodings_dict["input_ids"]))
            self.attn_masks.append(torch.tensor(encodings_dict["attention_mask"]))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return {
            "input_ids": self.input_ids[index],
            "attention_mask": self.attn_masks[index],
        }


#%%
# Preprocess data
str_list = get_processed_strings("data/original/", min_len=10)
# # max_length = max([len(tokenizer.encode(txt)) for txt in str_list])
max_length = 4188  # max length of the current dataset - no need to perform the previous line every time
if VERBOSE:
    print(f"The longest tirade is {max_length} tokens long.")


#%%
# Instantiate and split the dataset
TRAIN_PROP = 0.95

print("Initialising Dataset...")
dataset = CRDataset(str_list, tokenizer, max_length)

# Training and validation splits
train_size = int(TRAIN_PROP * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

if VERBOSE:
    print(f"Training size : {train_size}; Validation size : {val_size}")

#%%
# Instantiate model and adjust tokenizer
print("Fetching Model...")
config = GPT2Config.from_pretrained("gpt2")

model = GPT2LMHeadModel.from_pretrained("gpt2", config=config)
model.resize_token_embeddings(len(tokenizer))

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

#%%
# Instantiate Trainer

collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


training_args = TrainingArguments(
    output_dir="./logs/",
    per_device_train_batch_size=3,
    per_device_eval_batch_size=3,
    num_train_epochs=1,
    save_strategy="steps",
    save_total_limit=5,
    # evaluation_strategy="epoch",
    optim="adamw_torch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=collator,
)

#%%
# Training
trainer.train(resume_from_checkpoint=False)

#%%
