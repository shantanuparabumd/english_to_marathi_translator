import torch
from english_to_marathi_translator.models.transformer import Transformer
from english_to_marathi_translator.utils.path_utils import TOKENIZER_DIR, PROCESSED_DATA_DIR, PROJECT_ROOT
import sentencepiece as spm
import yaml


def load_config(config_file):
    """Load configuration from YAML file."""
    with open(config_file, "r") as file:
        return yaml.safe_load(file)


def load_checkpoint(filepath, model):
    """Load a model checkpoint."""
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Checkpoint loaded from {filepath}")


def translate(sentence, model, english_tokenizer, marathi_tokenizer, device, max_len=100):
    """Translate an English sentence into Marathi."""
    model.eval()
    with torch.no_grad():
        # Tokenize input sentence
        src_tokens = english_tokenizer.encode(sentence, out_type=int)
        src_tokens = torch.tensor(src_tokens, device=device).unsqueeze(0)

        # Prepare input and output sequences
        tgt_tokens = [marathi_tokenizer.piece_to_id("<BOS>")]  # Use BOS token
        marathi_vocab_size = marathi_tokenizer.get_piece_size()

        for _ in range(max_len):
            tgt_tensor = torch.tensor(tgt_tokens, device=device).unsqueeze(0)
            output = model(src_tokens, tgt_tensor)

            # Clamp token IDs to valid range
            next_token = output[0, -1].argmax().item()
            next_token = min(next_token, marathi_vocab_size - 1)  # Ensure token ID is within bounds
            tgt_tokens.append(next_token)

            if next_token == marathi_tokenizer.piece_to_id("<EOS>"):  # Stop on EOS token
                break

        # Decode predicted tokens, excluding BOS/EOS tokens
        valid_tokens = [
            t for t in tgt_tokens 
            if 0 <= t < marathi_vocab_size and t not in {marathi_tokenizer.piece_to_id("<BOS>"), marathi_tokenizer.piece_to_id("<EOS>")}
        ]
        translation = marathi_tokenizer.decode(valid_tokens)
        return translation



def infer():
    # Load configuration
    config = load_config(PROJECT_ROOT / "configs/default_config.yaml")

    # Load SentencePiece tokenizers
    english_tokenizer = spm.SentencePieceProcessor(model_file=str(TOKENIZER_DIR / "english.model"))
    marathi_tokenizer = spm.SentencePieceProcessor(model_file=str(TOKENIZER_DIR / "marathi.model"))

    # Initialize model
    device = torch.device(config["training"]["device"])
    model = Transformer(
        src_vocab_size=config["model"]["src_vocab_size"],
        tgt_vocab_size=config["model"]["tgt_vocab_size"],
        embed_size=config["model"]["embed_size"],
        num_layers=config["model"]["num_layers"],
        num_heads=config["model"]["num_heads"],
        ff_hidden=config["model"]["ff_hidden"],
        dropout=config["model"]["dropout"],
        max_len=config["model"]["max_len"],
        device=device,
    ).to(device)

    # Load best policy checkpoint
    best_policy_path = PROCESSED_DATA_DIR / "checkpoints_bharat/checkpoint_epoch19_batch500.pth"
    load_checkpoint(best_policy_path, model)

    # Test translations
    sample_sentences = [
        "Hello, how are you?",
        "What is your name?",
        "I am feeling good today.",
        "The weather is very pleasant."
    ]
    for sentence in sample_sentences:
        translation = translate(sentence, model, english_tokenizer, marathi_tokenizer, device)
        print(f"English: {sentence}")
        print(f"Marathi: {translation}")


if __name__ == "__main__":
    infer()
