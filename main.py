import torch as T
from preprocess import TOKENIZER
from models import CRTransformer
import numpy as np

# input phrase
n_w = 10
input_seq = "Hello everyone, and welcome to this episode of Critical"
tok_in = TOKENIZER(input_seq)

# get vocab
vocab = T.load("data/vocab.pth")
encoded_in = np.array([vocab[word] for word in tok_in])


# get saved model and load it
model = CRTransformer()
model.load_state_dict(T.load("data/last_model.pth", map_location="cpu"))
model.eval()

# predict words
pred_seq = ""
for i in range(n_w):
    if i == 0:
        src, tgt = T.from_numpy(encoded_in[:-1]), T.from_numpy(encoded_in[1:])

    _, out_indexes = model(src, tgt)

    src = tgt
    tgt = out_indexes.detach().clone()

    print(out_indexes)
    print(out_indexes[-1])
    pred_seq += vocab.get_itos()[out_indexes[-1]]

print(input_seq + " " + pred_seq)
