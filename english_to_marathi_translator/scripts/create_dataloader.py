import sentencepiece as spm
from torch.utils.data import DataLoader
from english_to_marathi_translator.utils.dataset import TranslationDataset, collate_fn
from english_to_marathi_translator.utils.path_utils import PROCESSED_DATA_DIR, TOKENIZER_DIR
import torch
from datasets import load_dataset

def tokenize_data(dataset_split, src_tokenizer, tgt_tokenizer, max_seq_length=100):
    tokenized_data = []
    for sample in dataset_split:
        src_tokens = src_tokenizer.encode(sample["translation"]["en"], out_type=int)
        tgt_tokens = tgt_tokenizer.encode(sample["translation"]["mr"], out_type=int)

        # Filter sequences longer than max_seq_length
        if len(src_tokens) <= max_seq_length and len(tgt_tokens) <= max_seq_length:
            tokenized_data.append((src_tokens, tgt_tokens))

    return tokenized_data

def create_dataloader(split="train", batch_size=32, max_seq_length=100):
    # Load the dataset
    dataset = load_dataset("opus100", "en-mr")[split]

    # Load tokenizers
    english_tokenizer = spm.SentencePieceProcessor(model_file=str(TOKENIZER_DIR / "english.model"))
    marathi_tokenizer = spm.SentencePieceProcessor(model_file=str(TOKENIZER_DIR / "marathi.model"))

    # Tokenize data
    tokenized_data = tokenize_data(dataset, english_tokenizer, marathi_tokenizer, max_seq_length)

    # Create DataLoader
    dataset = TranslationDataset(tokenized_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=(split == "train"))

    # Save DataLoader
    loader_path = PROCESSED_DATA_DIR / f"{split}_loader.pth"
    torch.save(dataloader, loader_path)
    print(f"{split.capitalize()} DataLoader saved at {loader_path}")

if __name__ == "__main__":
    create_dataloader(split="train", batch_size=32)
    create_dataloader(split="validation", batch_size=32)