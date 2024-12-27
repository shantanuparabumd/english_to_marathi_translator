from datasets import load_dataset

# Load the opus100 dataset for English-Marathi
dataset = load_dataset("opus100", "en-mr")

# Print some examples
print("Train sample:", dataset['train'][0])
print("Validation sample:", dataset['validation'][0])
print("Test sample:", dataset['test'][0])
print("Dataset structure:", dataset)

# Step 1: Open files for writing
with open("data/processed/english.txt", "w") as eng_file, \
     open("data/processed/marathi.txt", "w") as mar_file:

    # Step 2: Iterate through each sample in the training dataset
    for sample in dataset['train']:
        # Step 3: Extract English and Marathi translations
        english_sentence = sample['translation']['en']
        marathi_sentence = sample['translation']['mr']

        # Step 4: Write each sentence to the respective file
        eng_file.write(english_sentence + "\n")
        mar_file.write(marathi_sentence + "\n")