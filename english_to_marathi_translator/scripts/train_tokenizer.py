import sentencepiece as spm
from english_to_marathi_translator.utils.path_utils import PROCESSED_DATA_DIR, TOKENIZER_DIR

def train_sentencepiece_tokenizer(input_file, model_prefix, vocab_size=8000):
    """
    Train a SentencePiece tokenizer.

    Args:
        input_file (str): Path to the input text file.
        model_prefix (str): Prefix for the tokenizer model files.
        vocab_size (int): Vocabulary size for the tokenizer.
    """
    spm.SentencePieceTrainer.train(
        input=f"{input_file}",
        model_prefix=f"{model_prefix}",
        vocab_size=vocab_size,
        character_coverage=1.0,  # Adjust this for Marathi if necessary
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3
    )
    print(f"Tokenizer trained and saved as {model_prefix}.model and {model_prefix}.vocab")

def main():
    # Paths to input text files
    english_file = PROCESSED_DATA_DIR / "english.txt"
    marathi_file = PROCESSED_DATA_DIR / "marathi.txt"

    # Paths for output tokenizers
    english_model_prefix = TOKENIZER_DIR / "english"
    marathi_model_prefix = TOKENIZER_DIR / "marathi"

    # Train English tokenizer
    train_sentencepiece_tokenizer(input_file=english_file, model_prefix=english_model_prefix)

    # Train Marathi tokenizer
    train_sentencepiece_tokenizer(input_file=marathi_file, model_prefix=marathi_model_prefix)

if __name__ == "__main__":
    main()
