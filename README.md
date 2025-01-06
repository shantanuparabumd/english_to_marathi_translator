
# **English-to-Marathi Translation Using Transformers**

This project demonstrates the implementation of a Transformer-based machine translation model from scratch for translating English sentences to Marathi. The implementation includes the core components of the Transformer architecture such as Self-Attention, Multi-Head Attention, Positional Encoding, and Layer Normalization. The training process uses a parallel corpus of English-Marathi sentence pairs, and the model is evaluated using metrics like BLEU score.

---

## **Table of Contents**
1. [Introduction](#introduction)
2. [Theory](#theory)
   - [Self-Attention](#self-attention)
   - [Multi-Head Attention](#multi-head-attention)
   - [Positional Encoding](#positional-encoding)
   - [Layer Normalization](#layer-normalization)
   - [Feed-Forward Network](#feed-forward-network)
   - [Tokenization](#tokenization)
3. [Model Architecture](#model-architecture)
   - [Parameters and Hyperparameters](#parameters-and-hyperparameters)
4. [Datasets](#datasets)
5. [Training Details](#training-details)
6. [Evaluation Metrics](#evaluation-metrics)
7. [How to Recreate the Project](#how-to-recreate-the-project)
8. [Results](#results)
9. [References](#references)

---

## **Introduction**

Machine translation is a key task in Natural Language Processing (NLP) that focuses on automatically converting text from one language to another. In this project, we build a Transformer-based sequence-to-sequence model capable of translating English text into Marathi.

---

## **Theory**

### **1. Self-Attention**
The self-attention mechanism allows the model to dynamically focus on different parts of the input sequence when encoding or decoding. It computes attention scores between all pairs of tokens to capture dependencies.

**Equation**:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

### **2. Multi-Head Attention**
Multi-Head Attention extends self-attention by computing multiple attention heads in parallel. This enables the model to attend to different relationships in the input sequence.

### **3. Positional Encoding**
Since Transformers do not have recurrence or convolution, positional encoding is added to the input embeddings to inject information about the token's position in the sequence.

**Equation**:
$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right), \quad PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

### **4. Layer Normalization**
Layer normalization normalizes inputs to each layer, stabilizing training and improving convergence.

### **5. Feed-Forward Network**
Each Transformer layer includes a feed-forward network, enabling non-linear transformations to enhance learning.

---

## **Model Architecture**

The Transformer model includes the following components:
1. **Encoder**: Processes the source sequence with self-attention and feed-forward layers.
2. **Decoder**: Generates the target sequence using self-attention, cross-attention, and feed-forward layers.

### **Parameters and Hyperparameters**
- **d_model**: Embedding dimensionality (e.g., 512). Higher values improve model expressiveness but increase training time.
- **num_heads**: Number of attention heads. Determines the diversity of relationships the model captures.
- **ff_dim**: Dimensionality of the feed-forward network. Controls the model's capacity.
- **num_layers**: Number of encoder and decoder layers. More layers can capture deeper patterns but may lead to overfitting.

---

## **Datasets**

The project uses the [OPUS-100 dataset](https://huggingface.co/datasets/opus100), a multilingual corpus containing parallel sentence pairs. Specifically, the **English-Marathi** language pair is used for training and evaluation.

---

## **Training Details**

- **Optimization**: Adam optimizer with a learning rate of 0.0001.
- **Loss Function**: Cross-Entropy Loss, ignoring padding tokens.
- **Batch Size**: 16 sentences per batch.
- **Epochs**: 100 for convergence.

### **Steps to Train**:
1. Tokenize the input and target sentences.
2. Pad the sequences to equal lengths.
3. Generate causal masks for decoder inputs to prevent information leakage.
4. Pass the data through the Transformer model.
5. Compute loss and backpropagate.

---

## **Evaluation Metrics**

### **1. BLEU Score**
The **BLEU (Bilingual Evaluation Understudy)** score is used to measure the quality of the translations by comparing them to human translations. It ranges from 0 to 100, where higher is better.

### **2. Manual Inspection**
Sample translations are manually inspected for fluency and accuracy.

---

## **How to Recreate the Project**

### **Dependencies**
Install the required libraries:
```bash
pip install torch transformers datasets sacrebleu sentencepiece
```

### **Steps**
1. Preprocess the dataset to tokenize and pad sentences.
2. Train the model using the script provided.
3. Evaluate using BLEU score and inspect sample translations.

---

## **Results**

The model achieved a BLEU score of **22.63** on the test set. The translations are fluent and maintain semantic integrity for most sentences.

### **Sample Translations**
| **English**               | **True Marathi**          | **Predicted Marathi**     |
|---------------------------|---------------------------|---------------------------|
| Kenya used to be a British colony.       | केनिया ब्रिटिश वसाहत असायची.  | ब्रिटिशांनी एक दशलक्ष डॉलरचा शोध लावला.  |
| Tom will go, no matter what Mary says.              | टॉम जाईलच, मेरीने काहीही म्हटलं तरीही.         | टॉम म्हणतो की मेरी काहीही करणार नाही.         |
| He likes singing and dancing.     | त्यांना गायला व नाचायला आवडतं. | त्यांना नाचायला आणि नाचायला आवडतं. |

---

## **References**
- Vaswani, A., et al. (2017). [Attention Is All You Need](https://arxiv.org/abs/1706.03762).
- Hugging Face Transformers Library: https://huggingface.co/transformers
- OPUS-100 Dataset: https://huggingface.co/datasets/opus100
