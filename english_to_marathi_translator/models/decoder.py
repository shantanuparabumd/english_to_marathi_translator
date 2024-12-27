import torch
import torch.nn as nn
import torch.nn.functional as F
from english_to_marathi_translator.models.encoder import MultiHeadAttention, FeedForward

class DecoderLayer(nn.Module):
    def __init__(self, embed_size, num_heads, ff_hidden, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(embed_size, num_heads)
        self.cross_attention = MultiHeadAttention(embed_size, num_heads)
        self.feed_forward = FeedForward(embed_size, ff_hidden)

        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.norm3 = nn.LayerNorm(embed_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_out, src_mask, tgt_mask):
        self_attention = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attention))

        cross_attention = self.cross_attention(x, encoder_out, encoder_out, src_mask)
        x = self.norm2(x + self.dropout(cross_attention))

        forward = self.feed_forward(x)
        x = self.norm3(x + self.dropout(forward))

        return x

class Decoder(nn.Module):
    def __init__(self, tgt_vocab_size, embed_size, num_layers, num_heads, ff_hidden, dropout, max_len, device):
        super(Decoder, self).__init__()
        self.device = device
        self.embed_size = embed_size
        self.word_embedding = nn.Embedding(tgt_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_len, embed_size)

        self.layers = nn.ModuleList(
            [DecoderLayer(embed_size, num_heads, ff_hidden, dropout) for _ in range(num_layers)]
        )
        self.fc_out = nn.Linear(embed_size, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_out, src_mask, tgt_mask):
        batch_size, seq_length = x.shape

        # Ensure positions do not exceed positional embedding size
        positions = torch.arange(0, seq_length).clamp(max=self.position_embedding.num_embeddings - 1)
        positions = positions.expand(batch_size, seq_length).to(self.device)

        x = self.dropout(self.word_embedding(x) + self.position_embedding(positions))
        for layer in self.layers:
            x = layer(x, encoder_out, src_mask, tgt_mask)

        return self.fc_out(x)
