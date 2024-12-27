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
