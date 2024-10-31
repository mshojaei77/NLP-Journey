# Practical Tasks to Learn LLMs from Scratch

Large language models (LLMs) are revolutionizing the field of artificial intelligence, enabling remarkable advancements in natural language processing and beyond. This guide outlines a comprehensive, hands-on learning path for anyone interested in diving deep into the world of LLMs, from foundational concepts to cutting-edge techniques.

**This journey is structured into modules, each focusing on a specific aspect of LLMs, and includes practical tasks designed to solidify your understanding through implementation and experimentation.**

## Module I: Working with Text Data

- [x] **Tokenization Exploration:** Learn about tokenization in Natural Language Processing (NLP) and explore various tokenization techniques including whitespace, NLTK, SpaCy and Byte Pair Encoding. [**Open In Colab**](https://colab.research.google.com/github/mshojaei77/NLP-Journey/blob/main/ch1/Tokenization_BPE.ipynb)
- [x] **Hugging Face tokenizers:** Demonstrate how to use Hugging Face tokenizers to prepare text data for NLP models. [**Open In Colab**](https://colab.research.google.com/github/mshojaei77/NLP-Journey/blob/main/ch1/Hugging_Face_Tokenizers.ipynb)
- [x] **Training a new tokenizer from an old one:**  [Open In Colab](https://colab.research.google.com/drive/1452WFn66MZzYylTNcL6hV5Zd45sskzs7?usp=sharing)
- [x] **Custom Tokenizer Training:** Train custom tokenizers on custom text dataset with three different tokenizer models (BPE, WordPiece, and Unigram) [**Open In Colab**](https://colab.research.google.com/drive/1uYFoxwCKwshkchBgQ4y4z9cDfKRlwZ-e?usp=sharing)
- [x] **Tokenization in Large Language Models:** Build a Byte Pair Encoding (BPE) tokenizer from scratch, similar to the one used in OpenAI's GPT models base on Andrej Karpathy's [Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE) video.  [**Open In Colab**](https://colab.research.google.com/drive/1y0KnCFZvGVf_odSfcNAws6kcDD7HsI0L?usp=sharing)
- [ ] **N-gram Language Model Implementation:** Build a bigram language model using a text dataset (e.g., Project Gutenberg). Use the model to generate a short text sequence.
- [ ] **Word Embedding Training and Visualization:** Train word embeddings using Word2Vec, GloVe, and FastText on a text corpus (e.g., Wikipedia). Visualize the embeddings using t-SNE and explore word similarities and analogies.
- [ ] **Pre-trained Embedding Exploration:** Explore the properties of pre-trained word embeddings (e.g., GloVe) by performing word similarity and analogy tasks.
- [ ] **Neural Language Model Implementation:** Implement a simple neural language model using an RNN in PyTorch. Train it on a text dataset and evaluate its perplexity.
- [ ] **Masked Language Model Training:** Implement a masked language model training loop in PyTorch using a transformer encoder.

## Module II: LLM Architectures

- [ ] **Self-Attention Implementation:** Implement a basic self-attention mechanism in PyTorch. Visualize the attention weights to understand how the model attends to different parts of the input sequence.
- [ ] **Multi-Head Attention Comparison:** Implement multi-head attention and compare its performance with single-head attention on a sequence-to-sequence task (e.g., machine translation).
- [ ] **Transformer Encoder Implementation:** Implement a transformer encoder with positional encoding and feed-forward networks in PyTorch. Analyze the output representations.
- [ ] **Layer Normalization Experiment:** Explore different layer normalization techniques (LayerNorm, RMSNorm) in a transformer model. Compare their impact on training stability and performance.

## Module V: Pre-training Large Language Models
- [ ] **Data Collection:** Identify Data Sources, Gather Data, Ensure Data Diversity
- [ ] **Data Preprocessing:** Clean Data, Normalize Text, Tokenization
- [ ] **Simple Language Model Pre-training:** Pre-train a small language model on a text dataset (e.g., a subset of Wikipedia) using PyTorch. Evaluate its performance using perplexity.
- [ ] **Hyperparameter Optimization for Pre-training:** Experiment with different hyperparameters (learning rate, batch size) to optimize the pre-training performance of a small language model.
- [ ] **FLOPS Analysis:** Analyze the FLOPS (floating-point operations per second) of different pre-trained language models and compare their computational costs.

## Module VI: Fine-tuning LLMs 

- [ ] **GPT Fine-tuning for Text Generation:** Fine-tune a pre-trained GPT model for text generation using Hugging Face Transformers. Generate different text formats (e.g., stories, poems) using the fine-tuned model.
- [ ] **BERT Fine-tuning for Text Classification:** Fine-tune a pre-trained BERT model for text classification (e.g., sentiment analysis) using Hugging Face Transformers. Evaluate its performance on a benchmark dataset.
- [ ] **T5 Fine-tuning for Summarization:** Fine-tune a pre-trained T5 model for text summarization using Hugging Face Transformers. Evaluate the quality of the generated summaries using ROUGE scores.
- [ ] **Hugging Face Model Hub Exploration:** Explore and experiment with different pre-trained models from the Hugging Face Model Hub for tasks like question answering and translation.
- [ ] **LLM Fine-tuning for Text Classification:** Fine-tune a pre-trained LLM for text classification (e.g., topic classification) using a labeled dataset. Evaluate its performance using accuracy, precision, recall, and F1-score.
- [ ] **IMDB Movie Review Fine-tuning:** Fine-tune different pre-trained LLMs on the IMDB movie reviews dataset for sentiment analysis. Compare their performance and analyze the results.
- [ ] **Data Imbalance Handling:** Implement techniques for handling data imbalance (oversampling, undersampling) during fine-tuning of an LLM for a classification task with an imbalanced dataset.
- [ ] **Instruction Fine-tuning:** Fine-tune an LLM using an instruction dataset (e.g., Alpaca) and evaluate its performance on following instructions using appropriate metrics.
- [ ] **LoRA Implementation:** Implement LoRA for fine-tuning a pre-trained LLM on a downstream task. Compare its performance and efficiency with standard fine-tuning.
- [ ] **PEFT Techniques Comparison:** Implement and compare different PEFT techniques (LoRA, Adapters) on a specific task. Analyze their impact on memory footprint, training time, and performance.
- [ ] **Preference Dataset Creation:** Build a simple preference dataset for evaluating LLM outputs on a specific task (e.g., summarization). Collect human preferences for different model outputs.

## Module VIII: Optimizing LLM Training

- [ ] **Initialization and Optimizer Experiment:** Experiment with different initialization techniques (e.g., Xavier, Kaiming) and optimizers (Adam, AdamW) for training a small LLM. Analyze their impact on convergence speed and performance.
- [ ] **Learning Rate Scheduler Comparison:** Implement and compare different learning rate schedulers (cosine annealing, step decay) during LLM training. Analyze their impact on model performance.
- [ ] **Gradient Clipping and Accumulation Implementation:** Implement gradient clipping and accumulation in PyTorch during LLM training. Observe their effect on training stability and memory usage.
- [ ] **Mixed Precision Training with AMP:** Implement mixed precision training using PyTorch's automatic mixed precision (AMP) feature. Measure the speedup compared to full precision training.
- [ ] **Distributed Training with DDP:** Implement distributed training for a small LLM using PyTorch DDP (DistributedDataParallel) across multiple GPUs. Analyze the speedup and scaling efficiency.
- [ ] **Batch Size Experiment:** Experiment with different batch sizes during LLM training. Observe their impact on training time, memory usage, and model performance.

## Module IX: LLM Inference and Optimization

- [ ] **KV-Cache Implementation:** Implement KV-Cache for faster inference with a pre-trained LLM. Measure the speedup achieved by caching key-value pairs.
- [ ] **Model Quantization:** Implement model quantization using post-training quantization and quantization-aware training. Compare the quantized model's size, memory footprint, and performance with the original model.
- [ ] **Knowledge Distillation:** Implement knowledge distillation to train a smaller LLM from a larger pre-trained LLM. Evaluate the performance of the distilled model.
- [ ] **Model Pruning:** Implement model pruning using magnitude-based pruning or movement pruning. Analyze the impact on model size, inference speed, and performance.

## Module X: Deploying LLMs

- [ ] **Local LLM Deployment:** Deploy a pre-trained LLM on a local server using Flask or FastAPI. Create a simple interface to interact with the model.
- [ ] **Serverless LLM Deployment:** Deploy a pre-trained LLM as a serverless function using AWS Lambda or Google Cloud Functions. Create an API endpoint to access the model.
- [ ] **Performance Monitoring:** Implement a system for monitoring LLM performance in a simulated production environment. Track metrics like latency, throughput, and error rate.
- [ ] **Input Sanitization:** Implement input sanitization techniques to prevent prompt injection attacks on a deployed LLM.
- [ ] **Output Monitoring:** Implement output monitoring techniques to detect potentially harmful or inappropriate outputs from a deployed LLM.


## Module XI: Applications

- [ ] **Chatbot Development:** Build a simple chatbot using a pre-trained LLM and a web framework like Gradio or Streamlit.
- [ ] **Code Generation Assistant:** Implement a code generation assistant that can generate code snippets (e.g., Python, JavaScript) based on natural language descriptions.
- [ ] **Text Summarization Engine:** Build a text summarization engine using a pre-trained LLM. Evaluate the quality of the generated summaries using ROUGE scores.
- [ ] **Sentiment Analysis Implementation:** Implement a sentiment analysis system using both lexicon-based and machine learning approaches. Compare their performance on a sentiment analysis dataset.
- [ ] **Named Entity Recognition with NLTK/spaCy:** Implement a named entity recognition system using NLTK or spaCy. Evaluate its performance on a named entity recognition dataset.
- [ ] **Topic Modeling with LDA/NMF:** Implement a topic modeling system using LDA or NMF. Analyze the discovered topics from a collection of documents.
- [ ] **Text-to-Image Generation:** Implement a text-to-image generation system using a pre-trained diffusion model (e.g., Stable Diffusion). Generate images from various text prompts.
- [ ] **Video Understanding with Captions:** Implement a video understanding system that can generate captions or summaries for short videos using a pre-trained multimodal LLM.

## Module XII: Advanced Topics and Future Directions

- [ ] **CI/CD Pipeline for LLM Deployment:** Implement a CI/CD pipeline for automating the building, testing, and deployment of an LLM to a cloud platform (e.g., AWS, GCP).
- [ ] **Experiment Tracking System:** Implement a system for tracking and managing LLM experiments using tools like MLflow or Weights & Biases.
- [ ] **Capstone Project:** Build and deploy a real-world LLM application that addresses a specific problem or task. This could be anything from a specialized chatbot to a code generation tool tailored for a specific programming language.


