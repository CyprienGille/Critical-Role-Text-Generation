import torch as T
import torch.nn as nn
from torch.nn import Transformer
from math import log


class PositionalEncoding(nn.Module):
    """
    Provides addition of a positional encoding component
    """

    def __init__(self, dim_input, dropout=0.1, max_encodable_len=2**10):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = T.zeros(max_encodable_len, dim_input)
        positions = T.arange(0, max_encodable_len, dtype=T.float).unsqueeze(
            1
        )  # all possible positions, flat
        div_terms = T.exp(
            T.arange(0, dim_input, 2).float()
            * (-log(max_encodable_len * 2) / dim_input)
        )  # all frequencies
        pe[:, 0::2] = T.sin(positions * div_terms)  # even encodings
        pe[:, 1::2] = T.cos(positions * div_terms)  # uneven encodings
        pe = pe.unsqueeze(0).transpose(0, 1)  # so we can add it to the input
        self.register_buffer("pe", pe)  # the positional encodings are not a parameter

    def forward(self, x):
        # the forward pass is just an addition to the input sequence
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class CRTransformer(nn.Module):
    def __init__(
        self,
        vocab_size=52123,
        n_features=130,
        nheads=10,
        n_enc_layers=6,
        n_dec_layers=6,
        dim_feedforward=4096,
        dropout=0.1,
        device="cpu",
    ) -> None:
        super(CRTransformer, self).__init__()

        self.embedder = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=n_features
        )
        self.pos_encoder = PositionalEncoding(dim_input=n_features, dropout=dropout)

        # masks to prevent the model for seeing future words are provided by mask_check
        self.src_mask = None
        self.tgt_mask = None
        self.device = device

        self.transformer = Transformer(
            d_model=n_features,
            nhead=nheads,
            num_encoder_layers=n_enc_layers,
            num_decoder_layers=n_dec_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        self.debedder = nn.Linear(n_features, vocab_size)

        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        """Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
        """
        return self.transformer.generate_square_subsequent_mask(sz)

    def init_weights(self):
        # before training, init weights
        initrange = 0.5
        nn.init.uniform_(self.embedder.weight, -initrange, initrange)
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
                mask = self.generate_square_subsequent_mask(sz).to(self.device)
                self.src_mask = mask
        if src_or_tgt == "tgt":
            if self.tgt_mask is None or self.tgt_mask.size(0) != sz:
                mask = self.generate_square_subsequent_mask(sz).to(self.device)
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
        src = self.embedder(src)
        src = self.pos_encoder(src)
        tgt = self.embedder(tgt)
        tgt = self.pos_encoder(tgt)
        if src_masked:
            self.mask_check(len(src), "src")
        if tgt_masked:
            self.mask_check(len(tgt), "tgt")

        output = self.transformer(src, tgt, self.src_mask, self.tgt_mask)
        output = self.debedder(output)
        return output, T.argmax(output, dim=-1)
