import torch
import torch.nn as nn
from torch.nn import Transformer

## LOGSOFTMAX
import math


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PositionalEncoding(nn.Module):
    """Provides addition of a positional encoding component

    Extends:
        torch.nn.Module
    """

    def __init__(self, dim_input, dropout=0.1, max_encodable_len=10000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_encodable_len, dim_input)
        positions = torch.arange(0, max_encodable_len, dtype=torch.float).unsqueeze(
            1
        )  # all possible positions, flat
        div_terms = torch.exp(
            torch.arange(0, dim_input, 2).float()
            * (-math.log(max_encodable_len * 2) / dim_input)
        )  # all frequencies
        pe[:, 0::2] = torch.sin(positions * div_terms)  # even encodings
        pe[:, 1::2] = torch.cos(positions * div_terms)  # uneven encodings
        pe = pe.unsqueeze(0).transpose(0, 1)  # so we can add it to the input
        self.register_buffer("pe", pe)  # the positional encodings are not a parameter

    def forward(self, x):
        # the forward pass is just an addition to the input sequence
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class CRTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        dim_input=512,
        nb_heads=2,
        dim_ff=2048,
        nb_enc=6,
        nb_dec=6,
        dropout=0.5,
    ):
        """Creates a model made of : an embedder, a positional encoder, a transformer, a final linear layer
        using the parameters given. Initializes weights.

        Args:
            vocab_size {int} -- the number of diff tokens in the vocab (i.e. the starting dimension of the embedding)

        Keyword Args:
            dim_input {int} -- the dimension of the encoder and decoder inputs (default: {512})
            nb_heads {int} -- the number of heads of the MultiHeadAttention modules (default: {2})
            dim_ff {int} -- the dimension of the hidden feedforward networks in the Transfomer (default: {2048})
            nb_enc {int} -- the number of encoder layers (default: {6})
            nb_dec {int} -- the number of decoder layers (default: {6})
            dropout {float} -- the dropout value (default: {0.5})
        """
        super(CRTransformer, self).__init__()

        self.dim_input = dim_input
        self.embedder = nn.Embedding(vocab_size, dim_input)
        self.pos_encoder = PositionalEncoding(dim_input, dropout)

        # masks to prevent the model for seeing future words are provided by mask_check
        self.src_mask = None
        self.tgt_mask = None

        self.transformer = Transformer(
            d_model=dim_input,
            nhead=nb_heads,
            num_encoder_layers=nb_enc,
            num_decoder_layers=nb_dec,
            dim_feedforward=dim_ff,
            dropout=dropout,
        )

        self.debedder = nn.Linear(dim_input, vocab_size)

        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        """Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
        """
        return self.transformer.generate_square_subsequent_mask(sz)

    def init_weights(self):
        # before training, init weights
        initrange = 0.1
        nn.init.uniform_(self.embedder.weight, -initrange, initrange)
        nn.init.zeros_(self.debedder.weight)
        nn.init.uniform_(self.debedder.weight, -initrange, initrange)

    def mask_check(self, sz, src_or_tgt="src"):
        """performs mask existence and size checks (useful when processing end data (e.g. last batches of an epoch))

        Args:
            sz {int} -- the length of the sequence we intend to mask after

        Keyword Args:
            src_or_tgt {str} -- to toggle between checking the source or target mask (default: {"src"})
        """
        if src_or_tgt == "src":
            if self.src_mask is None or self.src_mask.size(0) != sz:
                mask = self.generate_square_subsequent_mask(sz).to(device)
                self.src_mask = mask
        if src_or_tgt == "tgt":
            if self.tgt_mask is None or self.tgt_mask.size(0) != sz:
                mask = self.generate_square_subsequent_mask(sz).to(device)
                self.tgt_mask = mask

    def forward(self, src, tgt, src_masked=False, tgt_masked=False):
        """NB: should respect the dimensions given in the pytorch documentation for the Transformer module

        Arguments:
            src -- [the encoder input]
            tgt -- [the decoder input, i.e. the model's last output]

        Keyword Arguments:
            src_masked {bool} -- [whether the source should be masked] (default: {False})
            tgt_masked {bool} -- [whether the target should be masked] (default: {False})

        Returns:
            [type] -- [description]
        """
        # nb: the math.sqrt(self.dim_input) is there to prevent dimension choice to matter, scaling the data
        src = self.embedder(src) * math.sqrt(self.dim_input)
        src = self.pos_encoder(src)
        tgt = self.embedder(tgt) * math.sqrt(self.dim_input)
        tgt = self.pos_encoder(tgt)
        if src_masked:
            self.mask_check(len(src), "src")
        if tgt_masked:
            self.mask_check(len(tgt), "tgt")

        output = self.transformer(src, tgt, self.src_mask, self.tgt_mask)
        output = self.debedder(output)
        return output
