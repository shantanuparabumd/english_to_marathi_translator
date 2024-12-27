from setuptools import setup, find_packages

setup(
    name="english_to_marathi_translator",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "datasets>=2.6.1",
        "sentencepiece>=0.1.99",
        "torch>=1.10.0"
    ],
    entry_points={
        "console_scripts": [
            "preprocess=english_to_marathi_translator.scripts.preprocess_data:main",
            "train_tokenizer=english_to_marathi_translator.scripts.train_tokenizer:main",
        ]
    },
    description="A Python package for English-to-Marathi translation using transformers",
    author="Shantanu Parab",
    author_email="shantanuparab99@gmail.com",
    python_requires=">=3.8",
)
