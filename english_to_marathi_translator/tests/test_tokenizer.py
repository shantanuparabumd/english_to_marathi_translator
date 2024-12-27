import sentencepiece as spm
from english_to_marathi_translator.utils.path_utils import TOKENIZER_DIR

# Load the tokenizers
english_tokenizer = spm.SentencePieceProcessor(model_file=str(TOKENIZER_DIR / "english.model"))
marathi_tokenizer = spm.SentencePieceProcessor(model_file=str(TOKENIZER_DIR / "marathi.model"))

# Test sentences
english_sentence = "Hello, how are you?"
marathi_sentence = "नमस्कार, आपण कसे आहात?"

# Tokenize sentences
english_tokens = english_tokenizer.encode(english_sentence, out_type=int)
marathi_tokens = marathi_tokenizer.encode(marathi_sentence, out_type=int)

# Decode tokens back to sentences
decoded_english = english_tokenizer.decode(english_tokens)
decoded_marathi = marathi_tokenizer.decode(marathi_tokens)

print("English tokens:", english_tokens)
print("Decoded English:", decoded_english)
print("Marathi tokens:", marathi_tokens)
print("Decoded Marathi:", decoded_marathi)
