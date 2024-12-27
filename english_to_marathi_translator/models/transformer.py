import torch
import torch.nn as nn
from english_to_marathi_translator.models.encoder import Encoder
from english_to_marathi_translator.models.decoder import Decoder

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, embed_size, num_layers, num_heads, ff_hidden, dropout, max_len, device):
        super(Transformer, self).__init__()
        self.encoder = Encoder(
            src_vocab_size=src_vocab_size,
            embed_size=embed_size,
            num_layers=num_layers,
            num_heads=num_heads,
            ff_hidden=ff_hidden,
            dropout=dropout,
            max_len=max_len,
            device=device,
        )
        self.decoder = Decoder(
            tgt_vocab_size=tgt_vocab_size,
            embed_size=embed_size,
            num_layers=num_layers,
            num_heads=num_heads,
            ff_hidden=ff_hidden,
            dropout=dropout,
            max_len=max_len,
            device=device,
        )

    def make_src_mask(self, src):
        """
        Create a mask for the source sequence to ignore padding tokens.
        """
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_length)
        return src_mask.to(src.device)

    def make_tgt_mask(self, tgt):
        """
        Create a mask for the target sequence to ignore padding tokens and future tokens.
        """
        seq_length = tgt.shape[1]
        tgt_mask = torch.tril(torch.ones((seq_length, seq_length))).expand(tgt.size(0), 1, seq_length, seq_length)
        return tgt_mask.to(tgt.device)

    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)

        encoder_out = self.encoder(src, src_mask)
        out = self.decoder(tgt, encoder_out, src_mask, tgt_mask)

        return out
