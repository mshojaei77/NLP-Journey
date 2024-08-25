## NLP Journey - Roadmap to Learn LLMs from Scratch with Modern NLP Methods in 2024

This repository provides a comprehensive guide for learning Natural Language Processing (NLP) from the ground up, progressing to the understanding and application of Large Language Models (LLMs). It focuses on practical skills needed for NLP and LLM-related roles in 2024 and beyond. We'll leverage Jupyter Notebooks for hands-on practice.

**Table of Contents**

* [Chapter 1: Foundations of NLP](#chapter-1-foundations-of-nlp)
    * [1.1 Introduction to NLP](#11-introduction-to-nlp)
    * [1.2 Text Preprocessing](#12-text-preprocessing)
    * [1.3 Feature Engineering](#13-feature-engineering)
    * [1.4 Word Embeddings](#14-word-embeddings)
* [Chapter 2: Essential NLP Tasks](#chapter-2-essential-nlp-tasks)
    * [2.1 Text Classification](#21-text-classification)
    * [2.2 Sentiment Analysis](#22-sentiment-analysis)
    * [2.3 Named Entity Recognition (NER)](#23-named-entity-recognition-ner)
    * [2.4 Topic Modeling](#24-topic-modeling)
* [Chapter 3: Deep Learning for NLP](#chapter-3-deep-learning-for-nlp)
    * [3.1 Neural Network Fundamentals](#31-neural-network-fundamentals)
    * [3.2 Deep Learning Frameworks](#32-deep-learning-frameworks)
    * [3.3 Deep Learning Architectures for NLP](#33-deep-learning-architectures-for-nlp)
* [Chapter 4: Large Language Models (LLMs)](#chapter-4-large-language-models-llms)
    * [4.1 The Transformer Architecture](#41-the-transformer-architecture)
    * [4.2 LLM Architectures](#42-llm-architectures)
    * [4.3 LLM Training](#43-llm-training)
    * [4.4 LLM Evaluation](#44-llm-evaluation)
    * [4.5 LLM Deployment](#45-llm-deployment)
* [Chapter 5: Multimodal Learning](#chapter-5-multimodal-learning)
    * [5.1 Multimodal LLMs](#51-multimodal-llms)
    * [5.2 Vision-Language Tasks](#52-vision-language-tasks)
    * [5.3 Multimodal Applications](#53-multimodal-applications)
* [Chapter 6: Deployment and Productionizing LLMs](#chapter-6-deployment-and-productionizing-llms)
    * [6.1 Deployment Strategies](#61-deployment-strategies)
    * [6.2 Inference Optimization](#62-inference-optimization)
    * [6.3 Building with LLMs](#63-building-with-llms)
    * [6.4 MLOps for LLMs](#64-mlops-for-llms)
    * [6.5 LLM Security](#65-llm-security)

---

# Chapter 1: Foundations of NLP

## 1.1 Introduction to NLP

* **What is NLP?** Explain the core concepts of Natural Language Processing, including:
    * **Syntax:** The arrangement of words and phrases to create well-formed sentences.
    * **Semantics:** The meaning of words, phrases, and sentences.
    * **Pragmatics:** How context contributes to meaning in language.
    * **Discourse:** How language is used in conversation and text to convey meaning beyond individual sentences.
* **Applications of NLP:** Provide a broad overview of how NLP is used in various domains.

| Category | Resources |
|---|---|
| Blog Tutorials | [What is Natural Language Processing (NLP)?](https://www.datacamp.com/blog/what-is-natural-language-processing) |

## 1.2 Text Preprocessing

* **Tokenization:**
    * **Word Tokenization:** Breaking text into individual words.
    * **Subword Tokenization:** Breaking words into smaller units (subwords), like Byte Pair Encoding (BPE) and SentencePiece. This helps handle out-of-vocabulary words.
* **Stemming:** Reducing words to their base or root form (e.g., "running" -> "run").
* **Lemmatization:** Converting words to their base form using vocabulary analysis (e.g., "better" -> "good").
* **Stop Word Removal:** Removing common words that carry less meaning (e.g., "the", "a", "is").
* **Punctuation Handling:** Removing or standardizing punctuation.

| Category | Resources |
|---|---|
| Blog Tutorials | [Stanford: Stemming and lemmatization](https://nlp.stanford.edu/IR-book/html/htmledition/stemming-and-lemmatization-1.html) |
| Notebooks | [Tokenization, Lemmatization, Stemming, and Sentence Segmentation](https://colab.research.google.com/drive/18ZnEnXKLQkkJoBXMZR2rspkWSm9EiDuZ) |
| Video Tutorials | [Andrej Karpathy: Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE&t=1158s) |
| Docs | [Hugging Face Tokenizers](https://huggingface.co/docs/tokenizers/index), [NLTK Stop Words](https://www.nltk.org/book/ch02.html#stop-words-corpus) |
| Code Examples | [NLTK Stemming and Lemmatization](https://www.nltk.org/howto/stem.html) |

## 1.3 Feature Engineering

* **Bag-of-Words (BoW):** Representing text as a collection of word frequencies.
* **TF-IDF:** A statistical measure that reflects how important a word is to a document in a collection.
* **N-grams:** Sequences of N consecutive words or characters.

| Category | Resources |
|---|---|
| Docs | [Scikit-learn: Text Feature Extraction](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) |

## 1.4 Word Embeddings

* **Word2Vec:** Learns vector representations of words based on their co-occurrence patterns in text.
* **GloVe:** Learns global vector representations of words by factoring a word-context co-occurrence matrix.
* **FastText:** An extension of Word2Vec that considers subword information, improving representations for rare words.
* **Contextual Embeddings:**
    * **ELMo:** Learns contextualized word representations by considering the entire sentence.
    * **BERT:** Uses a bidirectional transformer to generate deep contextualized word embeddings.

| Category | Resources |
|---|---|
| Blog Tutorials | [Jay Alammar - Illustrated Word2Vec](https://jalammar.github.io/illustrated-word2vec/), [Stanford NLP: N-gram Language Models](https://nlp.stanford.edu/fsnlp/lm.html) |
| Code Examples | [Gensim Word2Vec](https://radimrehurek.com/gensim/models/word2vec.html) |
| Papers | [Stanford GloVe](https://nlp.stanford.edu/projects/glove/) |

# Chapter 2: Essential NLP Tasks

## 2.1 Text Classification

* **What is Text Classification?**
* **Traditional Methods:**
    * Naive Bayes
    * SVM
    * Logistic Regression
* **Deep Learning Methods:**
    * Recurrent Neural Networks (RNNs)
    * Convolutional Neural Networks (CNNs)
    * Transformers

| Category | Resources |
|---|---|
| Tutorials | [Scikit-learn Text Classification](https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html) |
| Docs | [Hugging Face Text Classification](https://huggingface.co/docs/transformers/tasks/sequence_classification) |
| Code Examples | [FastText](https://github.com/facebookresearch/fastText) |

## 2.2 Sentiment Analysis

* **What is Sentiment Analysis?**
* **Lexicon-Based Approach:** Analyzing text for positive and negative words.
* **Machine Learning Approach:** Training models on labeled data to predict sentiment.
* **Aspect-Based Sentiment Analysis:** Identifying sentiment towards specific aspects of an entity.

| Category | Resources |
|---|---|
| Code Examples | [NLTK Sentiment Analysis](https://www.nltk.org/howto/sentiment.html), [TextBlob Sentiment Analysis](https://textblob.readthedocs.io/en/dev/quickstart.html#sentiment-analysis), [VADER Sentiment Analysis](https://github.com/cjhutto/vaderSentiment) |

## 2.3 Named Entity Recognition (NER)

* **What is NER?**
* **Rule-Based Systems:** Using patterns and rules to identify entities.
* **Machine Learning-Based Systems:** Training models to recognize entities.
* **Popular Tools:** NLTK, spaCy, Transformers

| Category | Resources |
|---|---|
| Docs | [Hugging Face NER](https://huggingface.co/docs/transformers/tasks/token-classification) |
| Code Examples | [NLTK NER](https://www.nltk.org/book/ch07.html), [spaCy NER](https://spacy.io/usage/linguistic-features#named-entities), [MIT Information Extraction Toolkit](https://github.com/mit-nlp/MITIE) |

## 2.4 Topic Modeling

* **What is Topic Modeling?**
* **Latent Dirichlet Allocation (LDA):** A probabilistic model for discovering latent topics in a collection of documents.
* **Non-Negative Matrix Factorization (NMF):** A linear algebra technique for topic modeling.

| Category | Resources |
|---|---|
| Code Examples | [Gensim Topic Modeling](https://radimrehurek.com/gensim/auto_examples/tutorials/run_lda.html), [Scikit-learn NMF](https://scikit-learn.org/stable/modules/decomposition.html#non-negative-matrix-factorization-nmf-or-nnmf), [BigARTM](https://github.com/bigartm/bigartm) |

# Chapter 3: Deep Learning for NLP

## 3.1 Neural Network Fundamentals

* **Neural Network Basics:**
    * Architecture of a neural network (layers, neurons, weights, biases).
    * Activation functions (sigmoid, ReLU, tanh).
* **Backpropagation:** The algorithm for training neural networks.
* **Gradient Descent:** Optimizing the weights of a neural network.
* **Vanishing Gradients:** Challenges in training deep neural networks for NLP.
* **Exploding Gradients:** Challenges in training deep neural networks for NLP.

| Category | Resources |
|---|---|
| Video Tutorials | [3Blue1Brown - Neural Networks](https://www.youtube.com/watch?v=aircAruvnKk), [freeCodeCamp - Deep Learning Crash Course](https://www.youtube.com/watch?v=VyWAvY2CF9c) |

## 3.2 Deep Learning Frameworks

* **PyTorch:**
* **TensorFlow:**
* **JAX:**
* **Considerations for Choosing a Framework:**
    * Ease of use
    * Community Support
    * Computational efficiency

| Category | Resources |
|---|---|
| Tutorials | [PyTorch Tutorials](https://pytorch.org/tutorials/), [TensorFlow Tutorials](https://www.tensorflow.org/tutorials) |
| Docs | [JAX Documentation](https://jax.readthedocs.io/en/latest/) |

## 3.3 Deep Learning Architectures for NLP

* **Recurrent Neural Networks (RNNs):**
    * Suitable for sequential data like text.
    * Types: LSTMs, GRUs.
* **Attention Mechanism:** Allows the network to focus on specific parts of the input sequence.
* **Convolutional Neural Networks (CNNs) for Text:**
    * Can capture local patterns in text.
    * Used for text classification and other tasks.
* **Sequence-to-Sequence Models:**
    * Used for tasks like machine translation and text summarization.
    * Encoder-decoder architecture.
* **Transformers:** The dominant architecture for sequence-to-sequence tasks, based on attention mechanisms.

| Category | Resources |
|---|---|
| Blog Tutorials | [colah's blog: Understanding LSTMs](https://colah.github.io/posts/2015-08-Understanding-LSTMs/), [Andrej Karpathy: The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/), [Understanding Convolutional Neural Networks for NLP](http://www.wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp/), [Jay Alammar: The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) |
| Blog Posts | [Google AI Blog: Transformer Networks](https://ai.googleblog.com/2017/08/transformer-networks-state-of-art.html) |

# Chapter 4: Large Language Models (LLMs)

## 4.1 The Transformer Architecture

* **Attention Mechanism:**
    * Self-attention
    * Multi-head attention.
    * **Scaled Dot-Product Attention:** The core attention mechanism used in Transformers.
* **Residual Connections:** Help train very deep networks.
* **Layer Normalization:** Improves training stability.
* **Positional Encodings:** Encoding the order of words in a sequence.

| Category | Resources |
|---|---|
| Blog Tutorials | [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/), [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/) |

## 4.2 LLM Architectures

* **Generative Pre-trained Transformer Models (GPT):** Autoregressive models, good at text generation.
* **Bidirectional Encoder Representations from Transformers (BERT):** Bidirectional models, excel at understanding context.
* **T5 (Text-to-Text Transfer Transformer):** A unified framework that treats all NLP tasks as text-to-text problems.
* **BART (Bidirectional and Auto-Regressive Transformers):** Combines the strengths of BERT and GPT for both understanding and generation.

| Category | Resources |
|---|---|
| Papers & Code | [LLMDataHub](https://github.com/Zjh-819/LLMDataHub) |

## 4.3 LLM Training

* **Pre-training:**
    * **Masked Language Modeling (MLM):** Predicting masked words in a sentence (used in BERT).
    * **Causal Language Modeling (CLM):** Predicting the next word in a sequence (used in GPT).
* **Post-training:**
    * **Domain Adaptation:** Adapting a pre-trained LLM to a specific domain or industry.
    * **Task-Specific Fine-Tuning:** Fine-tuning a pre-trained LLM on a specific downstream task with labeled data.
* **Fine-tuning:**
    * **Supervised Fine-Tuning (SFT):** Training on labeled data for a specific task.
* **Adapting LLMs:**
    * **Parameter-Efficient Fine-Tuning (PEFT):** Updating only a small subset of model parameters to reduce computational cost. Methods include:
        * **LoRA (Low-Rank Adaptation)**
        * **Adapters**
    * **Reinforcement Learning from Human Feedback (RLHF):** Using human feedback to train reward models and improve LLM alignment with human preferences.

| Category | Resources |
|---|---|
| Blog Tutorials | [Fine-Tune Your Own Llama 2 Model](https://mlabonne.github.io/blog/posts/Fine_Tune_Your_Own_Llama_2_Model_in_a_Colab_Notebook.html) |
| Blog Posts | [Hugging Face: Parameter-Efficient Fine-Tuning](https://huggingface.co/blog/peft) |
| Blog Tutorials & Code | [LoRA Insights](https://lightning.ai/pages/community/lora-insights/), [Distilabel](https://github.com/argilla-io/distilabel) |
| Blog Posts & Code Examples | [An Introduction to Training LLMs using RLHF](https://wandb.ai/ayush-thakur/Intro-RLAIF/reports/An-Introduction-to-Training-LLMs-Using-Reinforcement-Learning-From-Human-Feedback-RLHF---VmlldzozMzYyNjcy) |

## 4.4 LLM Evaluation

* **Evaluation Benchmarks:**


* **Evaluation Metrics:**

* **Prompt Engineering:**
    * **Zero-Shot Prompting:** Getting the model to perform a task without any task-specific training examples.
    * **Few-Shot Prompting:** Providing a few examples in the prompt to guide the model.
    * **Chain-of-Thought Prompting:** Encouraging the model to break down reasoning into steps.
    * **ReAct (Reason + Act):** Combining reasoning and action in prompts.

| Category | Resources |
|---|---|
| Guides & Tools | [Prompt Engineering Guide](https://www.promptingguide.ai/) |
| Blog Tutorials | [Lilian Weng: Prompt Engineering](https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/) |
| Papers | [Chain-of-Thoughts Papers](https://github.com/Timothyxxx/Chain-of-ThoughtsPapers) |

## 4.5 LLM Deployment

* **Deployment Strategies:**
    * **Local Servers:** Deploying LLMs on local machines for development and testing.
    * **Cloud Deployment:** Using cloud platforms like AWS, GCP, and Azure for scalable LLM deployment.
    * **Serverless Functions:** Deploying LLM inference as serverless functions for cost-effectiveness.
    * **Edge Deployment:** Running LLMs on edge devices like smartphones and IoT devices.

| Category | Resources |
|---|---|
| Tools | [LM Studio](https://lmstudio.ai/), [Ollama](https://ollama.ai/) |
| APIs & Platforms | [SkyPilot](https://github.com/skypilot-org/skypilot), [Hugging Face Inference API](https://huggingface.co/inference-api) |

# Chapter 5: Multimodal Learning

## 5.1 Multimodal LLMs

* **Learning from Multiple Modalities:** LLMs that can process and generate both text and other modalities, such as images, videos, and audio.
* **CLIP (Contrastive Language-Image Pretraining):** A model that learns joint representations of text and images.
* **ViT (Vision Transformer):** Applying the Transformer architecture to image data.
* **Other Multimodal Models:** Explore other architectures like LLaVA, MiniCPM-V, and GPT-SoVITS.

| Category | Resources |
|---|---|
| Papers | [OpenAI CLIP](https://openai.com/research/clip) |
| Blog Posts | [Google AI Blog: ViT](https://ai.googleblog.com/2020/10/an-image-is-worth-16x16-words.html) |

## 5.2 Vision-Language Tasks

* **Image Captioning:** Generating descriptions for images.
* **Visual Question Answering (VQA):** Answering questions about images.
* **Visual Reasoning:** Performing logical reasoning based on visual input.

| Category | Resources |
|---|---|
| Docs | [Hugging Face: Vision-Language Tasks](https://huggingface.co/docs/transformers/tasks/vision-language-modeling) |

## 5.3 Multimodal Applications

* **Text-to-Image Generation:** Generating images from text descriptions.
* **Video Understanding:** Analyzing and understanding video content.
* **Emerging Trends:**
    * **Neuro-Symbolic AI:** Combining neural networks with symbolic reasoning.
    * **LLMs for Robotics:** Using LLMs to control and interact with robots.

| Category | Resources |
|---|---|
| Models & Code | [Stability AI: Stable Diffusion](https://stability.ai/stable-image), [OpenAI DALL-E 2](https://openai.com/dall-e-2), [Hugging Face: Video Understanding](https://huggingface.co/docs/transformers/tasks/video-classification) |

# Chapter 6: Deployment and Productionizing LLMs

## 6.1 Deployment Strategies

* **Local Servers:** Deploying LLMs on local machines for development and testing.
* **Cloud Deployment:** Using cloud platforms like AWS, GCP, and Azure for scalable LLM deployment.
* **Serverless Functions:** Deploying LLM inference as serverless functions for cost-effectiveness.
* **Edge Deployment:** Running LLMs on edge devices like smartphones and IoT devices.

| Category | Resources |
|---|---|
| Tools | [LM Studio](https://lmstudio.ai/), [Ollama](https://ollama.ai/) |
| APIs & Platforms | [SkyPilot](https://github.com/skypilot-org/skypilot), [Hugging Face Inference API](https://huggingface.co/inference-api) |

## 6.2 Inference Optimization

* **Quantization:** Reducing the precision of model weights and activations to reduce memory footprint and speed up inference.
* **Flash Attention:** Optimizing the attention mechanism for faster and more efficient computation.
* **Knowledge Distillation:** Training smaller student models to mimic the behavior of larger teacher models.
* **Pruning:** Removing less important connections in the neural network.
* **Speculative Decoding:** Predicting future tokens during inference to speed up generation.

| Category | Resources |
|---|---|
| Blog Tutorials | [Introduction to Quantization](https://mlabonne.github.io/blog/posts/Introduction_to_Weight_Quantization.html) |
| Code Examples | [Flash-Attention](https://github.com/Dao-AILab/flash-attention) |

## 6.3 Building with LLMs

* **APIs:** Using pre-trained LLMs through APIs provided by OpenAI, Google, and others.
* **Web Frameworks:** Creating web applications that interact with LLMs using frameworks like Gradio and Streamlit.
* **User Interfaces:** Building graphical user interfaces for LLM applications.
* **Chatbots:** Building conversational interfaces powered by LLMs.

| Category | Resources |
|---|---|
| APIs & Platforms | [OpenAI API](https://platform.openai.com/), [Google AI Platform](https://cloud.google.com/ai-platform/), [Gradio](https://www.gradio.app/), [Streamlit](https://docs.streamlit.io/) |

## 6.4 MLOps for LLMs

* **CI/CD:** Continuous integration and continuous delivery pipelines for LLM development.
* **Monitoring:** Tracking LLM performance and detecting issues in production.
* **Model Management:** Managing different versions of LLM models and their deployments.
* **Experiment Tracking:** Tracking experiments, hyperparameters, and results during LLM development.
* **Data and Model Pipelines:** Building pipelines for data preprocessing, training, and deployment.

| Category | Resources |
|---|---|
| Tools | [CometLLM](https://github.com/comet-ml/comet-llm), [MLflow](https://mlflow.org/) |

## 6.5 LLM Security

* **Prompt Hacking:** Techniques to manipulate LLM behavior through malicious prompts.
    * **Prompt Injection**
    * **Prompt Leaking**
    * **Jailbreaking**
* **Backdoors:** Introducing vulnerabilities in LLMs during training. Methods include:
    * **Data Poisoning**
    * **Trigger Backdoors**
* **Defensive Measures:** Protecting LLMs from attacks and ensuring responsible use. Methods include:
    * **Red Teaming**
    * **Input Sanitization**
    * **Output Monitoring**

| Category | Resources |
|---|---|
| Guides & Cheat Sheets | [OWASP LLM Top 10](https://owasp.org/www-project-top-10-for-large-language-model-applications/), [Prompt Injection Primer](https://github.com/jthack/PIPE) |
