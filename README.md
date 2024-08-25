## NLP Journey - Roadmap to Learn LLMs from Scratch with Modern NLP Methods in 2024

This repository provides a comprehensive guide for learning Natural Language Processing (NLP) from the ground up, progressing to the understanding and application of Large Language Models (LLMs). It focuses on practical skills needed for NLP and LLM-related roles in 2024 and beyond. We'll leverage Jupyter Notebooks for hands-on practice.

**Table of Contents**

* [Chapter 1: Foundations of NLP](#chapter-1-foundations-of-nlp)
* [Chapter 2: Essential NLP Tasks](#chapter-2-essential-nlp-tasks)
* [Chapter 3: Deep Learning for NLP](#chapter-3-deep-learning-for-nlp)
* [Chapter 4: Large Language Models (LLMs)](#chapter-4-large-language-models-llms)
* [Chapter 5: Multimodal Learning](#chapter-5-multimodal-learning)
* [Chapter 6: Deployment and Productionizing LLMs](#chapter-6-deployment-and-productionizing-llms)
---

# Chapter 1: Foundations of NLP

## 1.1 Introduction to NLP

* **What is NLP?** Explain the core concepts of Natural Language Processing, including:
    * **Syntax:** The arrangement of words and phrases to create well-formed sentences.
    * **Semantics:** The meaning of words, phrases, and sentences.
    * **Pragmatics:** How context contributes to meaning in language.
    * **Discourse:** How language is used in conversation and text to convey meaning beyond individual sentences.
* **Applications of NLP:** Provide a broad overview of how NLP is used in various domains.

| Resources | Description |
|---|---|
| [Understanding Natural Language Processing](https://dev.to/avinashvagh/understanding-the-concept-of-natural-language-processing-nlp-and-prompt-engineering-35hg) | A comprehensive overview of NLP, its history, techniques, and applications, including prompt engineering. |
| [What is NLP?](https://www.secoda.co/glossary/what-is-nlp-natural-language-processing) | Detailed insights into NLP's workings, challenges, and its impact on technology interactions. |
| [NLP Definition from TechTarget](https://www.techtarget.com/searchenterpriseai/definition/natural-language-processing-NLP) | An explanation of NLP's history, techniques, and its applications in various fields. |
| [IBM's Overview of NLP](https://www.ibm.com/topics/natural-language-processing) | Discusses NLP's role in AI, its applications in business, and the importance of model selection. |
| [Introduction to NLP by TextMine](https://textmine.com/post/an-introduction-to-natural-language-processing) | An introduction to NLP's key aspects and its applications across industries. |

## 1.2 Text Preprocessing

Text preprocessing is a crucial step in Natural Language Processing (NLP) that prepares raw text data for analysis and model training. This section outlines modern techniques and resources relevant to text preprocessing as of 2024.

* **Tokenization:**
    * **Word Tokenization:** Breaking text into individual words, which is essential for various NLP tasks.
    * **Subword Tokenization:** Dividing words into smaller units (subwords) using methods like Byte Pair Encoding (BPE) and SentencePiece. This technique is particularly useful for handling out-of-vocabulary words and improving model performance.
* **Stemming:** Reducing words to their base or root form (e.g., "running" becomes "run"). This method simplifies the text but may not always yield valid words.
* **Lemmatization:** Converting words to their base form using vocabulary analysis (e.g., "better" becomes "good"). Unlike stemming, lemmatization ensures that the resulting words are valid and contextually appropriate.
* **Stop Word Removal:** Eliminating common words that carry less meaning (e.g., "the", "a", "is") to reduce noise in the data.
* **Punctuation Handling:** Standardizing or removing punctuation to ensure consistency in the text data.
* **Normalization:** Converting text to a standard format, such as lowercasing, removing extra spaces, and correcting misspellings, to improve the quality of the data.
* **Text Augmentation:** Techniques such as synonym replacement, random insertion, or back-translation to artificially expand the training dataset and improve model robustness.

| Resources | Description |
|---|---|
| [Neptune.ai: Tokenization in NLP](https://neptune.ai/blog/tokenization-in-nlp) | Comprehensive overview of tokenization types, challenges, and tools available in the NLP community. |
| [Stanford: Stemming and lemmatization](https://nlp.stanford.edu/IR-book/html/htmledition/stemming-and-lemmatization-1.html) | Explains stemming and lemmatization techniques. |
| [Tokenization, Lemmatization, Stemming, and Sentence Segmentation](https://colab.research.google.com/drive/18ZnEnXKLQkkJoBXMZR2rspkWSm9EiDuZ) | Practical notebook for tokenization, lemmatization, and stemming. |
| [Andrej Karpathy: Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE&t=1158s) | Video tutorial on building a tokenizer for GPT. |
| [Tokenmonster GitHub Repository](https://github.com/tokenmonster) | A new tokenization method aimed at improving the performance of large language models by optimizing token representation. |
| [WandB: An introduction to tokenization](https://wandb.ai/mostafaibrahim17/ml-articles/reports/An-introduction-to-tokenization-in-natural-language-processing--Vmlldzo3NTM4MzE5) | A guide exploring essential tokenization techniques and their applications in NLP. |
| [NLTK Stop Words Documentation](https://www.nltk.org/book/ch02.html#stop-words-corpus) | NLTK resource for handling stop words. |
| [NLTK Stemming and Lemmatization Documentation](https://www.nltk.org/howto/stem.html) | Code examples for stemming and lemmatization using NLTK. |
| [Hugging Face Tokenizers Documentation](https://huggingface.co/docs/tokenizers/index) | Detailed documentation on tokenization tools and libraries, emphasizing modern approaches in NLP. |
| [spaCy Documentation](https://spacy.io/usage/linguistic-features#tokenization) | Official guide on using spaCy for tokenization and other NLP tasks, highlighting its efficiency and capabilities. |

## 1.3 Feature Engineering

* **Bag-of-Words (BoW):** Representing text as a collection of word frequencies.
* **TF-IDF:** A statistical measure that reflects how important a word is to a document in a collection.
* **N-grams:** Sequences of N consecutive words or characters.

| Resources | Description |
|---|---|
| [Introduction to the Bag-of-Words (BoW) Model - PyImageSearch](https://pyimagesearch.com/2022/07/04/introduction-to-the-bag-of-words-bow-model/) | A comprehensive guide explaining the Bag-of-Words model, its implementation, pros and cons, and practical applications in natural language processing. |
| [A Quick Introduction to Bag of Words and TF-IDF](https://dataknowsall.com/blog/bowtfidf.html) | This article introduces the Bag-of-Words model and TF-IDF, detailing how they are used in text processing and machine learning, along with practical coding examples. |
| [N-grams Made Simple & How To Implement In Python (NLTK) - Spot Intelligence](https://spotintelligence.com/n-grams-made-simple-how-to-implement-in-python-nltk/) | An easy-to-follow resource that explains N-grams, their significance in NLP, and how to implement them using Python's NLTK library. |
| [NLP Basics: Tokens, N-Grams, and Bag-of-Words Models - Zilliz blog](https://zilliz.com/learn/introduction-to-natural-language-processing-tokens-ngrams-bag-of-words-models) | This blog post covers the fundamentals of NLP, including tokens, N-grams, and Bag-of-Words models, providing insights into their applications and limitations. |
| [Scikit-learn: Text Feature Extraction](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) | Documentation on text feature extraction methods. |

## 1.4 Word Embeddings

* **Word2Vec:** Learns vector representations of words based on their co-occurrence patterns in text.
* **GloVe:** Learns global vector representations of words by factoring a word-context co-occurrence matrix.
* **FastText:** An extension of Word2Vec that considers subword information, improving representations for rare words.
* **Contextual Embeddings:**
    * **ELMo:** Learns contextualized word representations by considering the entire sentence.
    * **BERT:** Uses a bidirectional transformer to generate deep contextualized word embeddings.

| Resources | Description |
|---|---|
| [Jay Alammar - Illustrated Word2Vec](https://jalammar.github.io/illustrated-word2vec/) | Visual explanation of Word2Vec. |
| [Stanford NLP: N-gram Language Models](https://nlp.stanford.edu/fsnlp/lm.html) | Overview of N-gram language models. |
| [Gensim Word2Vec](https://radimrehurek.com/gensim/models/word2vec.html) | Code examples for Word2Vec using Gensim. |
| [Stanford GloVe](https://nlp.stanford.edu/projects/glove/) | Paper and resources on GloVe embeddings. |

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

| Resources | Description |
|---|---|
| [Scikit-learn Text Classification](https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html) | Tutorial on text classification using scikit-learn. |
| [Hugging Face Text Classification](https://huggingface.co/docs/transformers/tasks/sequence_classification) | Documentation on text classification with Hugging Face. |
| [FastText](https://github.com/facebookresearch/fastText) | Code examples and resources for FastText. |

## 2.2 Sentiment Analysis

* **What is Sentiment Analysis?**
* **Lexicon-Based Approach:** Analyzing text for positive and negative words.
* **Machine Learning Approach:** Training models on labeled data to predict sentiment.
* **Aspect-Based Sentiment Analysis:** Identifying sentiment towards specific aspects of an entity.

| Resources | Description |
|---|---|
| [NLTK Sentiment Analysis](https://www.nltk.org/howto/sentiment.html) | Code examples for sentiment analysis using NLTK. |
| [TextBlob Sentiment Analysis](https://textblob.readthedocs.io/en/dev/quickstart.html#sentiment-analysis) | Quickstart guide for sentiment analysis with TextBlob. |
| [VADER Sentiment Analysis](https://github.com/cjhutto/vaderSentiment) | Code and documentation for VADER sentiment analysis. |

## 2.3 Named Entity Recognition (NER)

* **What is NER?**
* **Rule-Based Systems:** Using patterns and rules to identify entities.
* **Machine Learning-Based Systems:** Training models to recognize entities.
* **Popular Tools:** NLTK, spaCy, Transformers

| Resources | Description |
|---|---|
| [Hugging Face NER](https://huggingface.co/docs/transformers/tasks/token-classification) | Documentation on NER with Hugging Face. |
| [NLTK NER](https://www.nltk.org/book/ch07.html) | NLTK resources for NER. |
| [spaCy NER](https://spacy.io/usage/linguistic-features#named-entities) | spaCy documentation on NER. |
| [MIT Information Extraction Toolkit](https://github.com/mit-nlp/MITIE) | Toolkit for information extraction, including NER. |

## 2.4 Topic Modeling

* **What is Topic Modeling?**
* **Latent Dirichlet Allocation (LDA):** A probabilistic model for discovering latent topics in a collection of documents.
* **Non-Negative Matrix Factorization (NMF):** A linear algebra technique for topic modeling.

| Resources | Description |
|---|---|
| [Gensim Topic Modeling](https://radimrehurek.com/gensim/auto_examples/tutorials/run_lda.html) | Tutorial on topic modeling with Gensim. |
| [Scikit-learn NMF](https://scikit-learn.org/stable/modules/decomposition.html#non-negative-matrix-factorization-nmf-or-nnmf) | Documentation on NMF with scikit-learn. |
| [BigARTM](https://github.com/bigartm/bigartm) | Code and resources for advanced topic modeling. |

# Chapter 3: Deep Learning for NLP

## 3.1 Neural Network Fundamentals

* **Neural Network Basics:**
    * Architecture of a neural network (layers, neurons, weights, biases).
    * Activation functions (sigmoid, ReLU, tanh).
* **Backpropagation:** The algorithm for training neural networks.
* **Gradient Descent:** Optimizing the weights of a neural network.
* **Vanishing Gradients:** Challenges in training deep neural networks for NLP.
* **Exploding Gradients:** Challenges in training deep neural networks for NLP.

| Resources | Description |
|---|---|
| [3Blue1Brown - Neural Networks](https://www.youtube.com/watch?v=aircAruvnKk) | Visual introduction to neural networks. |
| [freeCodeCamp - Deep Learning Crash Course](https://www.youtube.com/watch?v=VyWAvY2CF9c) | Comprehensive crash course on deep learning. |

## 3.2 Deep Learning Frameworks

* **PyTorch:**
* **TensorFlow:**
* **JAX:**
* **Considerations for Choosing a Framework:**
    * Ease of use
    * Community Support
    * Computational efficiency

| Resources | Description |
|---|---|
| [PyTorch Tutorials](https://pytorch.org/tutorials/) | Official PyTorch tutorials. |
| [TensorFlow Tutorials](https://www.tensorflow.org/tutorials) | Official TensorFlow tutorials. |
| [JAX Documentation](https://jax.readthedocs.io/en/latest/) | Documentation for JAX. |

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

| Resources | Description |
|---|---|
| [colah's blog: Understanding LSTMs](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) | In-depth explanation of LSTMs. |
| [Andrej Karpathy: The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) | Blog post on the effectiveness of RNNs. |
| [Understanding Convolutional Neural Networks for NLP](http://www.wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp/) | Introduction to CNNs for NLP. |
| [Jay Alammar: The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) | Visual explanation of the Transformer architecture. |
| [Google AI Blog: Transformer Networks](https://ai.googleblog.com/2017/08/transformer-networks-state-of-art.html) | Overview of Transformer networks. |

# Chapter 4: Large Language Models (LLMs)

## 4.1 The Transformer Architecture

* **Attention Mechanism:**
    * Self-attention
    * Multi-head attention.
    * **Scaled Dot-Product Attention:** The core attention mechanism used in Transformers.
* **Residual Connections:** Help train very deep networks.
* **Layer Normalization:** Improves training stability.
* **Positional Encodings:** Encoding the order of words in a sequence.

| Resources | Description |
|---|---|
| [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) | Visual introduction to the Transformer architecture. |
| [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/) | Visual explanation of GPT-2. |

## 4.2 LLM Architectures

* **Generative Pre-trained Transformer Models (GPT):** Autoregressive models, good at text generation.
* **Bidirectional Encoder Representations from Transformers (BERT):** Bidirectional models, excel at understanding context.
* **T5 (Text-to-Text Transfer Transformer):** A unified framework that treats all NLP tasks as text-to-text problems.
* **BART (Bidirectional and Auto-Regressive Transformers):** Combines the strengths of BERT and GPT for both understanding and generation.

| Resources | Description |
|---|---|
| [LLMDataHub](https://github.com/Zjh-819/LLMDataHub) | Repository with papers and code on LLMs. |

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

| Resources | Description |
|---|---|
| [Fine-Tune Your Own Llama 2 Model](https://mlabonne.github.io/blog/posts/Fine_Tune_Your_Own_Llama_2_Model_in_a_Colab_Notebook.html) | Tutorial on fine-tuning a Llama 2 model. |
| [Hugging Face: Parameter-Efficient Fine-Tuning](https://huggingface.co/blog/peft) | Blog post on parameter-efficient fine-tuning. |
| [LoRA Insights](https://lightning.ai/pages/community/lora-insights/) | Insights into LoRA for parameter-efficient fine-tuning. |
| [Distilabel](https://github.com/argilla-io/distilabel) | Code and resources for distillation and labeling. |
| [An Introduction to Training LLMs using RLHF](https://wandb.ai/ayush-thakur/Intro-RLAIF/reports/An-Introduction-to-Training-LLMs-Using-Reinforcement-Learning-From-Human-Feedback-RLHF---VmlldzozMzYyNjcy) | Introduction to RLHF for training LLMs. |

## 4.4 LLM Evaluation

* **Evaluation Benchmarks:**
* **Evaluation Metrics:**
* **Prompt Engineering:**
    * **Zero-Shot Prompting:** Getting the model to perform a task without any task-specific training examples.
    * **Few-Shot Prompting:** Providing a few examples in the prompt to guide the model.
    * **Chain-of-Thought Prompting:** Encouraging the model to break down reasoning into steps.
    * **ReAct (Reason + Act):** Combining reasoning and action in prompts.

| Resources | Description |
|---|---|
| [Prompt Engineering Guide](https://www.promptingguide.ai/) | Comprehensive guide on prompt engineering. |
| [Lilian Weng: Prompt Engineering](https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/) | Blog post on prompt engineering. |
| [Chain-of-Thoughts Papers](https://github.com/Timothyxxx/Chain-of-ThoughtsPapers) | Collection of papers on chain-of-thought prompting. |

## 4.5 LLM Deployment

* **Deployment Strategies:**
    * **Local Servers:** Deploying LLMs on local machines for development and testing.
    * **Cloud Deployment:** Using cloud platforms like AWS, GCP, and Azure for scalable LLM deployment.
    * **Serverless Functions:** Deploying LLM inference as serverless functions for cost-effectiveness.
    * **Edge Deployment:** Running LLMs on edge devices like smartphones and IoT devices.

| Resources | Description |
|---|---|
| [LM Studio](https://lmstudio.ai/) | Tool for local LLM deployment. |
| [Ollama](https://ollama.ai/) | Tool for local LLM deployment. |
| [SkyPilot](https://github.com/skypilot-org/skypilot) | Tool for cloud deployment of LLMs. |
| [Hugging Face Inference API](https://huggingface.co/inference-api) | API for deploying LLMs. |

# Chapter 5: Multimodal Learning

## 5.1 Multimodal LLMs

* **Learning from Multiple Modalities:** LLMs that can process and generate both text and other modalities, such as images, videos, and audio.
* **CLIP (Contrastive Language-Image Pretraining):** A model that learns joint representations of text and images.
* **ViT (Vision Transformer):** Applying the Transformer architecture to image data.
* **Other Multimodal Models:** Explore other architectures like LLaVA, MiniCPM-V, and GPT-SoVITS.

| Resources | Description |
|---|---|
| [OpenAI CLIP](https://openai.com/research/clip) | Paper on CLIP model. |
| [Google AI Blog: ViT](https://ai.googleblog.com/2020/10/an-image-is-worth-16x16-words.html) | Blog post on Vision Transformer. |

## 5.2 Vision-Language Tasks

* **Image Captioning:** Generating descriptions for images.
* **Visual Question Answering (VQA):** Answering questions about images.
* **Visual Reasoning:** Performing logical reasoning based on visual input.

| Resources | Description |
|---|---|
| [Hugging Face: Vision-Language Tasks](https://huggingface.co/docs/transformers/tasks/vision-language-modeling) | Documentation on vision-language tasks. |

## 5.3 Multimodal Applications

* **Text-to-Image Generation:** Generating images from text descriptions.
* **Video Understanding:** Analyzing and understanding video content.
* **Emerging Trends:**
    * **Neuro-Symbolic AI:** Combining neural networks with symbolic reasoning.
    * **LLMs for Robotics:** Using LLMs to control and interact with robots.

| Resources | Description |
|---|---|
| [Stability AI: Stable Diffusion](https://stability.ai/stable-image) | Model for text-to-image generation. |
| [OpenAI DALL-E 2](https://openai.com/dall-e-2) | Model for text-to-image generation. |
| [Hugging Face: Video Understanding](https://huggingface.co/docs/transformers/tasks/video-classification) | Documentation on video understanding. |

# Chapter 6: Deployment and Productionizing LLMs

## 6.1 Deployment Strategies

* **Local Servers:** Deploying LLMs on local machines for development and testing.
* **Cloud Deployment:** Using cloud platforms like AWS, GCP, and Azure for scalable LLM deployment.
* **Serverless Functions:** Deploying LLM inference as serverless functions for cost-effectiveness.
* **Edge Deployment:** Running LLMs on edge devices like smartphones and IoT devices.

| Resources | Description |
|---|---|
| [LM Studio](https://lmstudio.ai/) | Tool for local LLM deployment. |
| [Ollama](https://ollama.ai/) | Tool for local LLM deployment. |
| [SkyPilot](https://github.com/skypilot-org/skypilot) | Tool for cloud deployment of LLMs. |
| [Hugging Face Inference API](https://huggingface.co/inference-api) | API for deploying LLMs. |

## 6.2 Inference Optimization

* **Quantization:** Reducing the precision of model weights and activations to reduce memory footprint and speed up inference.
* **Flash Attention:** Optimizing the attention mechanism for faster and more efficient computation.
* **Knowledge Distillation:** Training smaller student models to mimic the behavior of larger teacher models.
* **Pruning:** Removing less important connections in the neural network.
* **Speculative Decoding:** Predicting future tokens during inference to speed up generation.

| Resources | Description |
|---|---|
| [Introduction to Quantization](https://mlabonne.github.io/blog/posts/Introduction_to_Weight_Quantization.html) | Blog tutorial on quantization. |
| [Flash-Attention](https://github.com/Dao-AILab/flash-attention) | Code examples for Flash Attention. |

## 6.3 Building with LLMs

* **APIs:** Using pre-trained LLMs through APIs provided by OpenAI, Google, and others.
* **Web Frameworks:** Creating web applications that interact with LLMs using frameworks like Gradio and Streamlit.
* **User Interfaces:** Building graphical user interfaces for LLM applications.
* **Chatbots:** Building conversational interfaces powered by LLMs.

| Resources | Description |
|---|---|
| [OpenAI API](https://platform.openai.com/) | API for OpenAI models. |
| [Google AI Platform](https://cloud.google.com/ai-platform/) | Platform for deploying Google AI models. |
| [Gradio](https://www.gradio.app/) | Framework for building web interfaces with LLMs. |
| [Streamlit](https://docs.streamlit.io/) | Framework for building web applications with LLMs. |

## 6.4 MLOps for LLMs

* **CI/CD:** Continuous integration and continuous delivery pipelines for LLM development.
* **Monitoring:** Tracking LLM performance and detecting issues in production.
* **Model Management:** Managing different versions of LLM models and their deployments.
* **Experiment Tracking:** Tracking experiments, hyperparameters, and results during LLM development.
* **Data and Model Pipelines:** Building pipelines for data preprocessing, training, and deployment.

| Resources | Description |
|---|---|
| [CometLLM](https://github.com/comet-ml/comet-llm) | Tool for experiment tracking and model management. |
| [MLflow](https://mlflow.org/) | Open-source platform for the machine learning lifecycle. |

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

| Resources | Description |
|---|---|
| [OWASP LLM Top 10](https://owasp.org/www-project-top-10-for-large-language-model-applications/) | Guide on LLM security risks. |
| [Prompt Injection Primer](https://github.com/jthack/PIPE) | Primer on prompt injection attacks. |
