# Working with Text Data
## Tokenizing Text
- Learn about tokenization in Natural Language Processing (NLP) and explore various tokenization techniques including whitespace, NLTK, SpaCy and Byte Pair Encoding. [**Open In Colab**](https://colab.research.google.com/github/mshojaei77/NLP-Journey/blob/main/ch1/Tokenization_BPE.ipynb)
- Demonstrate how to use Hugging Face tokenizers to prepare text data for NLP models. [**Open In Colab**](https://colab.research.google.com/github/mshojaei77/NLP-Journey/blob/main/ch1/Hugging_Face_Tokenizers.ipynb)
- Training a new tokenizer from an old one [**Open In Colab**](https://colab.research.google.com/drive/1452WFn66MZzYylTNcL6hV5Zd45sskzs7?usp=sharing)
- Train custom tokenizers on custom text dataset with three different tokenizer models (BPE, WordPiece, and Unigram) [**Open In Colab**](https://colab.research.google.com/drive/1uYFoxwCKwshkchBgQ4y4z9cDfKRlwZ-e?usp=sharing)
- Build a Byte Pair Encoding (BPE) tokenizer from scratch, similar to the one used in OpenAI's GPT models base on Andrej Karpathy's [Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE) video.  [**Open In Colab**](https://colab.research.google.com/drive/1y0KnCFZvGVf_odSfcNAws6kcDD7HsI0L?usp=sharing)

## Embeddings
- Understanding Word Embeddings for Text Data [**Open In Colab**](https://colab.research.google.com/github/mshojaei77/LLMs-Journey/blob/main/ch1/Word_Embeddings.ipynb)
- Visualizing Embeddings using Dimensionality Reduction (t-SNE, PCA)
- Comparing Different Embedding Techniques (Word2Vec vs GloVe vs FastText)
- Using Pre-trained Embeddings
- Embedding Layers in Large Language Models (LLMs)
- Sentence/Paragraph/Document embeddings
- Fine-tuning Embeddings for Domain-Specific Tasks
- Cross-lingual Embeddings and Multilingual Applications
- Evaluating Embedding Quality and Performance Metrics

---
- Attention Mechanisms
    - Modeling Long Sequences: Challenges
    - Capturing Data Dependencies with Attention Mechanisms
    - Self-Attention
        - A Simple Self-Attention Mechanism
        - Computing Attention Weights
    - Self-Attention with Trainable Weights
        - Computing Attention Weights Step-by-Step
        - Implementing Self-Attention in Python
    - Causal Attention
        - Applying a Causal Attention Mask
        - Masking with Dropout
        - Implementing Causal Attention in Python
    - Multi-Head Attention
        - Stacking Single-Head Attention Layers
        - Implementing Multi-Head Attention

- Implementing a GPT Model for Text Generation
    - Coding an LLM Architecture
    - Layer Normalization
    - Feed Forward Network with GELU Activations
    - Shortcut Connections
    - Connecting Attention and Linear Layers
    - The GPT Model Code
    - Generating Text

- Pretraining on Unlabeled Data
    - Evaluating Generative Text Models
        - Generating Text with GPT
        - Calculating Text Generation Loss
        - Calculating Training and Validation Loss
    - Training an LLM
    - Decoding Strategies
        - Temperature Scaling
        - Top-k Sampling
        - Modifying Text Generation
    - Saving and Loading Model Weights
    - Loading Pretrained Weights from OpenAI

- Fine-tuning for Classification
    - Categories of Fine-tuning
    - Data Preparation
    - Creating Data Loaders
    - Initializing with Pretrained Weights
    - Adding a Classification Head
    - Calculating Loss and Accuracy
    - Fine-tuning on Supervised Data
    - Spam Classification Example

- Fine-tuning for Instruction Following
    - Introduction to Instruction Fine-tuning
    - Preparing Data
    - Organizing Data into Batches
    - Creating Data Loaders
    - Loading a Pretrained LLM
    - Fine-tuning on Instruction Data
    - Extracting and Saving Responses
    - Evaluating the Fine-tuned LLM
    - Conclusions
        - Next Steps
        - Staying Up-to-Date
        - Final Words
