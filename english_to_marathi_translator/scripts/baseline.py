import torch
from transformers import MarianMTModel, MarianTokenizer
from datasets import load_dataset
import sacrebleu

# Function to load a pre-trained MarianMT model and tokenizer for English-Marathi
def load_translation_model():
    model_name = "Helsinki-NLP/opus-mt-en-mr"  # Pre-trained model for English to Marathi
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device

# Function to translate a batch of sentences
def translate_sentences(sentences, tokenizer, model, device):
    inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        translated_tokens = model.generate(**inputs, max_length=128)
    translated_sentences = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
    return translated_sentences

# Load a dataset for benchmarking
def load_custom_dataset():
    """
    Load the English-Marathi subset of the opus100 dataset for benchmarking.
    """
    dataset = load_dataset("opus100", "en-mr", split="test[:10%]")  # Load 10% of the test set
    english_sentences = [item["translation"]["en"] for item in dataset]
    marathi_sentences = [item["translation"]["mr"] for item in dataset]
    return english_sentences, marathi_sentences


# Compute BLEU score
def compute_bleu(reference, prediction):
    bleu = sacrebleu.corpus_bleu(prediction, [reference])
    return bleu.score

# Compute other potential metrics (if needed)
def compute_additional_metrics(reference, prediction):
    # Placeholder for additional metrics like METEOR, TER, etc.
    # Implement here if desired
    pass

# Main function for benchmarking
def main():
    # Load model and tokenizer
    tokenizer, model, device = load_translation_model()

    # Load custom dataset for evaluation
    print("Loading dataset...")
    english_sentences, marathi_sentences = load_custom_dataset()

    # Translate the dataset
    print("Translating sentences...")
    batch_size = 16
    predicted_sentences = []
    for i in range(0, len(english_sentences), batch_size):
        batch = english_sentences[i:i + batch_size]
        translations = translate_sentences(batch, tokenizer, model, device)
        predicted_sentences.extend(translations)

    # Compute BLEU score
    print("\n--- Evaluation Metrics ---")
    bleu_score = compute_bleu(marathi_sentences, predicted_sentences)
    print(f"BLEU Score: {bleu_score:.2f}")

    # Display sample translations
    print("\n--- Sample Translations ---")
    for eng, true_mar, pred_mar in zip(english_sentences[:5], marathi_sentences[:5], predicted_sentences[:5]):
        print(f"English: {eng}")
        print(f"True Marathi: {true_mar}")
        print(f"Predicted Marathi: {pred_mar}\n")

if __name__ == "__main__":
    main()
