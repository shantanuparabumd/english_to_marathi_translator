import torch
from nltk.translate.bleu_score import corpus_bleu
from english_to_marathi_translator.models.transformer import Transformer
from english_to_marathi_translator.utils.path_utils import PROCESSED_DATA_DIR, TOKENIZER_DIR, PROJECT_ROOT
from english_to_marathi_translator.utils.dataset import TranslationDataset, collate_fn
import sentencepiece as spm
import yaml


def load_config(config_file):
    """Load configuration from YAML file."""
    with open(config_file, "r") as file:
        return yaml.safe_load(file)


def load_checkpoint(filepath, model, optimizer=None):
    """Load a model checkpoint."""
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    print(f"Checkpoint loaded from {filepath}")


def evaluate():
    # Load the config file
    config = load_config(PROJECT_ROOT / "configs/default_config.yaml")

    # Load the validation data
    val_loader = torch.load(PROCESSED_DATA_DIR / "validation_loader.pth")

    # Load SentencePiece tokenizers
    english_tokenizer = spm.SentencePieceProcessor(model_file=str(TOKENIZER_DIR / "english.model"))
    marathi_tokenizer = spm.SentencePieceProcessor(model_file=str(TOKENIZER_DIR / "marathi.model"))

    # Get tokenizer vocab size
    marathi_vocab_size = marathi_tokenizer.get_piece_size()

    # Initialize model using dimensions from config
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

    # Evaluation mode
    model.eval()
    references = []
    hypotheses = []

    with torch.no_grad():
        for src, tgt in val_loader:
            src, tgt = src.to(device), tgt.to(device)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            # Model prediction
            output = model(src, tgt_input)
            pred_tokens = output.argmax(dim=-1)

            # Decode predictions and references
            for pred, ref in zip(pred_tokens, tgt):
                # Filter out invalid token IDs
                valid_pred = [token for token in pred.cpu().numpy().tolist() if 0 <= token < marathi_vocab_size]
                valid_ref = [token for token in ref.cpu().numpy().tolist() if 0 <= token < marathi_vocab_size]

                pred_sentence = marathi_tokenizer.decode(valid_pred)
                ref_sentence = marathi_tokenizer.decode(valid_ref)

                references.append([ref_sentence.split()])
                hypotheses.append(pred_sentence.split())

    # Calculate BLEU score
    bleu_score = corpus_bleu(references, hypotheses)
    print(f"Validation BLEU Score: {bleu_score:.4f}")


if __name__ == "__main__":
    evaluate()
