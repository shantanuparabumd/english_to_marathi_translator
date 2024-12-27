import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert embed_size % num_heads == 0, "Embed size must be divisible by number of heads"
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads

        # Query, Key, Value linear layers
        self.q_linear = nn.Linear(embed_size, embed_size)
        self.k_linear = nn.Linear(embed_size, embed_size)
        self.v_linear = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]

        # Linear transformation and split into heads
        Q = self.q_linear(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        energy = torch.matmul(Q, K.transpose(-1, -2)) / (self.head_dim ** 0.5)
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        attention = torch.softmax(energy, dim=-1)

        # Aggregate values
        out = torch.matmul(attention, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_size)

        # Final linear layer
        return self.fc_out(out)

class FeedForward(nn.Module):
    def __init__(self, embed_size, ff_hidden):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_size, ff_hidden)
        self.fc2 = nn.Linear(ff_hidden, embed_size)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

class EncoderLayer(nn.Module):
    def __init__(self, embed_size, num_heads, ff_hidden, dropout):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(embed_size, num_heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = FeedForward(embed_size, ff_hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # Self-attention and residual connection
        attention = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attention))

        # Feed-forward and residual connection
        forward = self.feed_forward(x)
        x = self.norm2(x + self.dropout(forward))

        return x

class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embed_size, num_layers, num_heads, ff_hidden, dropout, max_len, device):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_len, embed_size)
        self.layers = nn.ModuleList(
            [EncoderLayer(embed_size, num_heads, ff_hidden, dropout) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        batch_size, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(batch_size, seq_length).to(self.device)
        x = self.dropout(self.word_embedding(x) + self.position_embedding(positions))

        for layer in self.layers:
            x = layer(x, mask)

        return x

if __name__ == "__main__":
    # Dummy input
    src_vocab_size = 10000
    embed_size = 512
    num_layers = 6
    num_heads = 8
    ff_hidden = 2048
    dropout = 0.1
    max_len = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = Encoder(src_vocab_size, embed_size, num_layers, num_heads, ff_hidden, dropout, max_len, device).to(device)
    dummy_input = torch.randint(0, src_vocab_size, (32, 50)).to(device)  # (batch_size, sequence_length)
    dummy_mask = None  # No mask for now
    output = encoder(dummy_input, dummy_mask)
    print(output.shape)  # Expected: (32, 50, 512)
