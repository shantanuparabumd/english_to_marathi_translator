from english_to_marathi_translator.utils.path_utils import PROCESSED_DATA_DIR
from datasets import load_dataset
from pathlib import Path

def main():
    # Load the dataset with the 'mr' config
    dataset = load_dataset("ai4bharat/samanantar", "mr")

    # Ensure processed directory exists
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Save text files for training tokenizers
    english_file = PROCESSED_DATA_DIR / "english.txt"
    marathi_file = PROCESSED_DATA_DIR / "marathi.txt"

    with open(english_file, "w") as eng_file, open(marathi_file, "w") as mar_file:
        for sample in dataset["train"]:
            # Use correct keys for English (src) and Marathi (tgt) translations
            eng_file.write(sample["src"] + "\n")
            mar_file.write(sample["tgt"] + "\n")
    
    print(f"Saved English sentences to {english_file}")
    print(f"Saved Marathi sentences to {marathi_file}")

if __name__ == "__main__":
    main()
