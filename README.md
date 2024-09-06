# 🚀 Practical Guide to Learning Large Language Models (LLMs) with Modern NLP Methods in 2024 🚀

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

* **[Introduction to Natural Language Processing and fundamental linguistic concepts](https://github.com/mshojaei77/NLP-Journey/blob/main/ch1/nlp-intro.md)** 

<sub>Sources: [Basics of Linguistics](https://www.ai.uni-hannover.de/fileadmin/ai/teaching/inlp-23s/part02-linguistics.pdf) </sub>

## 1.2 Text Preprocessing

#### Tokenization:
Breaking down text into smaller, meaningful units called tokens.
| Topic | Materials |
|--|--|
| Introduction to Tokenization, Byte Pair Encoding (BPE) | <a target="_blank" href="https://colab.research.google.com/github/mshojaei77/NLP-Journey/blob/main/ch1/Tokenization_BPE.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| Hugging Face Tokenizers | <a target="_blank" href="https://colab.research.google.com/github/mshojaei77/NLP-Journey/blob/main/ch1/Hugging_Face_Tokenizers.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| Train a Tokenizer with Hugging Face Tokenizers | <a target="_blank" href="https://colab.research.google.com/drive/1tPIPqKL_pUrxTst-7X59-EyXVCK-z6Rm?usp=sharing"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |

<sub>Sources: [Hugging Face Tokenizers Documentation](https://huggingface.co/docs/tokenizers/index), [ Everything About Tokenization](https://github.com/SumanthRH/tokenization), [Andrej Karpathy: Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE&t=1158s) </sub>

#### Stemming and Lemmatization:
* **Stemming:** Reducing words to their base or root form (e.g., "running" becomes "run"). This method simplifies the text but may not always yield valid words.
* **Lemmatization:** Converting words to their base form using vocabulary analysis (e.g., "better" becomes "good"). Unlike stemming, lemmatization ensures that the resulting words are valid and contextually appropriate.
 
* <sub>Sources:  [Tokenization, Lemmatization, Stemming, and Sentence Segmentation](https://colab.research.google.com/drive/18ZnEnXKLQkkJoBXMZR2rspkWSm9EiDuZ), [Stanford: Stemming and lemmatization](https://nlp.stanford.edu/IR-book/html/htmledition/stemming-and-lemmatization-1.html),[NLTK Stemming and Lemmatization Documentation](https://www.nltk.org/howto/stem.html)</sub>

#### Stop Word Removal
Eliminating common words that carry less meaning (e.g., "the", "a", "is") to reduce noise in the data.

<sub>Sources: [NLTK Stop Words Documentation](https://www.nltk.org/book/ch02.html#stop-words-corpus)</sub>


#### Punctuation Handling:
Standardizing or removing punctuation to ensure consistency in the text data.
  
#### Normalization:
Converting text to a standard format, such as lowercasing, removing extra spaces, and correcting misspellings, to improve the quality of the data.

#### Text Augmentation:
Techniques such as synonym replacement, random insertion, or back-translation to artificially expand the training dataset and improve model robustness.

## 1.3 Feature Engineering

* **Bag-of-Words (BoW):** Representing text as a collection of word frequencies.
* **TF-IDF:** A statistical measure that reflects how important a word is to a document in a collection.
* **N-grams:** Sequences of N consecutive words or characters.

<sub>Sources: [Introduction to the Bag-of-Words (BoW) Model - PyImageSearch](https://pyimagesearch.com/2022/07/04/introduction-to-the-bag-of-words-bow-model/), [A Quick Introduction to Bag of Words and TF-IDF](https://dataknowsall.com/blog/bowtfidf.html), [N-grams Made Simple & How To Implement In Python (NLTK) - Spot Intelligence](https://spotintelligence.com/n-grams-made-simple-how-to-implement-in-python-nltk/), [NLP Basics: Tokens, N-Grams, and Bag-of-Words Models - Zilliz blog](https://zilliz.com/learn/introduction-to-natural-language-processing-tokens-ngrams-bag-of-words-models), [Scikit-learn: Text Feature Extraction](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction)</sub>

## 1.4 Word Embeddings

* **Word2Vec:** Learns vector representations of words based on their co-occurrence patterns in text.
* **GloVe:** Learns global vector representations of words by factoring a word-context co-occurrence matrix.
* **FastText:** An extension of Word2Vec that considers subword information, improving representations for rare words.
* **Contextual Embeddings:**
    * **ELMo:** Learns contextualized word representations by considering the entire sentence.
    * **BERT:** Uses a bidirectional transformer to generate deep contextualized word embeddings.

<sub>Sources: [Jay Alammar - Illustrated Word2Vec](https://jalammar.github.io/illustrated-word2vec/), [Stanford NLP: N-gram Language Models](https://nlp.stanford.edu/fsnlp/lm.html), [Gensim Word2Vec](https://radimrehurek.com/gensim/models/word2vec.html), [Stanford GloVe](https://nlp.stanford.edu/projects/glove/)</sub>

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

<sub>Sources: [Scikit-learn Text Classification](https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html), [Hugging Face Text Classification](https://huggingface.co/docs/transformers/tasks/sequence_classification), [FastText](https://github.com/facebookresearch/fastText)</sub>


## 2.2 Sentiment Analysis

* **What is Sentiment Analysis?**
* **Lexicon-Based Approach:** Analyzing text for positive and negative words.
* **Machine Learning Approach:** Training models on labeled data to predict sentiment.
* **Aspect-Based Sentiment Analysis:** Identifying sentiment towards specific aspects of an entity.

<sub>Sources: [NLTK Sentiment Analysis](https://www.nltk.org/howto/sentiment.html), [TextBlob Sentiment Analysis](https://textblob.readthedocs.io/en/dev/quickstart.html#sentiment-analysis), [VADER Sentiment Analysis](https://github.com/cjhutto/vaderSentiment)</sub>


## 2.3 Named Entity Recognition (NER)

* **What is NER?**
* **Rule-Based Systems:** Using patterns and rules to identify entities.
* **Machine Learning-Based Systems:** Training models to recognize entities.
* **Popular Tools:** NLTK, spaCy, Transformers

<sub>Sources: [Hugging Face NER](https://huggingface.co/docs/transformers/tasks/token-classification), [NLTK NER](https://www.nltk.org/book/ch07.html), [spaCy NER](https://spacy.io/usage/linguistic-features#named-entities), [MIT Information Extraction Toolkit](https://github.com/mit-nlp/MITIE)</sub>


## 2.4 Topic Modeling

* **What is Topic Modeling?**
* **Latent Dirichlet Allocation (LDA):** A probabilistic model for discovering latent topics in a collection of documents.
* **Non-Negative Matrix Factorization (NMF):** A linear algebra technique for topic modeling.

<sub>Sources: [Gensim Topic Modeling](https://radimrehurek.com/gensim/auto_examples/tutorials/run_lda.html), [Scikit-learn NMF](https://scikit-learn.org/stable/modules/decomposition.html#non-negative-matrix-factorization-nmf-or-nnmf), [BigARTM](https://github.com/bigartm/bigartm)</sub>

# Chapter 3: Deep Learning for NLP

## 3.1 Neural Network Fundamentals

* **Neural Network Basics:**
    * Architecture of a neural network (layers, neurons, weights, biases).
    * Activation functions (sigmoid, ReLU, tanh).
* **Backpropagation:** The algorithm for training neural networks.
* **Gradient Descent:** Optimizing the weights of a neural network.
* **Vanishing Gradients:** Challenges in training deep neural networks for NLP.
* **Exploding Gradients:** Challenges in training deep neural networks for NLP.

<sub>Sources: [3Blue1Brown - Neural Networks](https://www.youtube.com/watch?v=aircAruvnKk), [freeCodeCamp - Deep Learning Crash Course](https://www.youtube.com/watch?v=VyWAvY2CF9c)</sub>

## 3.2 Deep Learning Frameworks

* **PyTorch:**
* **TensorFlow:**
* **JAX:**
* **Considerations for Choosing a Framework:**
    * Ease of use
    * Community Support
    * Computational efficiency

<sub>Sources: [PyTorch Tutorials](https://pytorch.org/tutorials/), [TensorFlow Tutorials](https://www.tensorflow.org/tutorials), [JAX Documentation](https://jax.readthedocs.io/en/latest/)</sub>

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

<sub>Sources: [colah's blog: Understanding LSTMs](https://colah.github.io/posts/2015-08-Understanding-LSTMs/), [Andrej Karpathy: The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/), [Understanding Convolutional Neural Networks for NLP](http://www.wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp/), [Jay Alammar: The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/), [Google AI Blog: Transformer Networks](https://ai.googleblog.com/2017/08/transformer-networks-state-of-art.html)</sub>

# Chapter 4: Large Language Models (LLMs)

## 4.1 The Transformer Architecture

* **Attention Mechanism:**
    * Self-attention
    * Multi-head attention.
    * **Scaled Dot-Product Attention:** The core attention mechanism used in Transformers.
* **Residual Connections:** Help train very deep networks.
* **Layer Normalization:** Improves training stability.
* **Positional Encodings:** Encoding the order of words in a sequence.

<sub>Sources: [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/), [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/)</sub>

## 4.2 LLM Architectures

* **Generative Pre-trained Transformer Models (GPT):** Autoregressive models, good at text generation.
* **Bidirectional Encoder Representations from Transformers (BERT):** Bidirectional models, excel at understanding context.
* **T5 (Text-to-Text Transfer Transformer):** A unified framework that treats all NLP tasks as text-to-text problems.
* **BART (Bidirectional and Auto-Regressive Transformers):** Combines the strengths of BERT and GPT for both understanding and generation.

<sub>Sources: [LLMDataHub](https://github.com/Zjh-819/LLMDataHub)</sub>

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

<sub>Sources: [Fine-Tune Your Own Llama 2 Model](https://mlabonne.github.io/blog/posts/Fine_Tune_Your_Own_Llama_2_Model_in_a_Colab_Notebook.html), [Hugging Face: Parameter-Efficient Fine-Tuning](https://huggingface.co/blog/peft), [LoRA Insights](https://lightning.ai/pages/community/lora-insights/), [Distilabel](https://github.com/argilla-io/distilabel), [An Introduction to Training LLMs using RLHF](https://wandb.ai/ayush-thakur/Intro-RLAIF/reports/An-Introduction-to-Training-LLMs-Using-Reinforcement-Learning-From-Human-Feedback-RLHF---VmlldzozMzYyNjcy)</sub>

## 4.4 LLM Evaluation

* **Evaluation Benchmarks:**
* **Evaluation Metrics:**
* **Prompt Engineering:**
    * **Zero-Shot Prompting:** Getting the model to perform a task without any task-specific training examples.
    * **Few-Shot Prompting:** Providing a few examples in the prompt to guide the model.
    * **Chain-of-Thought Prompting:** Encouraging the model to break down reasoning into steps.
    * **ReAct (Reason + Act):** Combining reasoning and action in prompts.

<sub>Sources: [Prompt Engineering Guide](https://www.promptingguide.ai/), [Lilian Weng: Prompt Engineering](https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/), [Chain-of-Thoughts Papers](https://github.com/Timothyxxx/Chain-of-ThoughtsPapers)</sub>

## 4.5 LLM Deployment

* **Deployment Strategies:**
    * **Local Servers:** Deploying LLMs on local machines for development and testing.
    * **Cloud Deployment:** Using cloud platforms like AWS, GCP, and Azure for scalable LLM deployment.
    * **Serverless Functions:** Deploying LLM inference as serverless functions for cost-effectiveness.
    * **Edge Deployment:** Running LLMs on edge devices like smartphones and IoT devices.

<sub>Sources: [LM Studio](https://lmstudio.ai/), [Ollama](https://ollama.ai/), [SkyPilot](https://github.com/skypilot-org/skypilot), [Hugging Face Inference API](https://huggingface.co/inference-api)</sub>

# Chapter 5: Multimodal Learning

## 5.1 Multimodal LLMs

* **Learning from Multiple Modalities:** LLMs that can process and generate both text and other modalities, such as images, videos, and audio.
* **CLIP (Contrastive Language-Image Pretraining):** A model that learns joint representations of text and images.
* **ViT (Vision Transformer):** Applying the Transformer architecture to image data.
* **Other Multimodal Models:** Explore other architectures like LLaVA, MiniCPM-V, and GPT-SoVITS.

<sub>Sources: [OpenAI CLIP](https://openai.com/research/clip), [Google AI Blog: ViT](https://ai.googleblog.com/2020/10/an-image-is-worth-16x16-words.html)</sub>

## 5.2 Vision-Language Tasks

* **Image Captioning:** Generating descriptions for images.
* **Visual Question Answering (VQA):** Answering questions about images.
* **Visual Reasoning:** Performing logical reasoning based on visual input.

<sub>Sources: [Hugging Face: Vision-Language Tasks](https://huggingface.co/docs/transformers/tasks/vision-language-modeling)</sub>

## 5.3 Multimodal Applications

* **Text-to-Image Generation:** Generating images from text descriptions.
* **Video Understanding:** Analyzing and understanding video content.
* **Emerging Trends:**
    * **Neuro-Symbolic AI:** Combining neural networks with symbolic reasoning.
    * **LLMs for Robotics:** Using LLMs to control and interact with robots.

<sub>Sources: [Stability AI: Stable Diffusion](https://stability.ai/stable-image), [OpenAI DALL-E 2](https://openai.com/dall-e-2), [Hugging Face: Video Understanding](https://huggingface.co/docs/transformers/tasks/video-classification)</sub>

# Chapter 6: Deployment and Productionizing LLMs

## 6.1 Deployment Strategies

* **Local Servers:** Deploying LLMs on local machines for development and testing.
* **Cloud Deployment:** Using cloud platforms like AWS, GCP, and Azure for scalable LLM deployment.
* **Serverless Functions:** Deploying LLM inference as serverless functions for cost-effectiveness.
* **Edge Deployment:** Running LLMs on edge devices like smartphones and IoT devices.

<sub>Sources: [LM Studio](https://lmstudio.ai/), [Ollama](https://ollama.ai/), [SkyPilot](https://github.com/skypilot-org/skypilot), [Hugging Face Inference API](https://huggingface.co/inference-api)</sub>

## 6.2 Inference Optimization

* **Quantization:** Reducing the precision of model weights and activations to reduce memory footprint and speed up inference.
* **Flash Attention:** Optimizing the attention mechanism for faster and more efficient computation.
* **Knowledge Distillation:** Training smaller student models to mimic the behavior of larger teacher models.
* **Pruning:** Removing less important connections in the neural network.
* **Speculative Decoding:** Predicting future tokens during inference to speed up generation.

<sub>Sources: [Introduction to Quantization](https://mlabonne.github.io/blog/posts/Introduction_to_Weight_Quantization.html), [Flash-Attention](https://github.com/Dao-AILab/flash-attention)</sub>

## 6.3 Building with LLMs

* **APIs:** Using pre-trained LLMs through APIs provided by OpenAI, Google, and others.
* **Web Frameworks:** Creating web applications that interact with LLMs using frameworks like Gradio and Streamlit.
* **User Interfaces:** Building graphical user interfaces for LLM applications.
* **Chatbots:** Building conversational interfaces powered by LLMs.

<sub>Sources: [OpenAI API](https://platform.openai.com/), [Google AI Platform](https://cloud.google.com/ai-platform/), [Gradio](https://www.gradio.app/), [Streamlit](https://docs.streamlit.io/)</sub>

## 6.4 MLOps for LLMs

* **CI/CD:** Continuous integration and continuous delivery pipelines for LLM development.
* **Monitoring:** Tracking LLM performance and detecting issues in production.
* **Model Management:** Managing different versions of LLM models and their deployments.
* **Experiment Tracking:** Tracking experiments, hyperparameters, and results during LLM development.
* **Data and Model Pipelines:** Building pipelines for data preprocessing, training, and deployment.

<sub>Sources: [CometLLM](https://github.com/comet-ml/comet-llm), [MLflow](https://mlflow.org/)</sub>

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

<sub>Sources: [OWASP LLM Top 10](https://owasp.org/www-project-top-10-for-large-language-model-applications/), [Prompt Injection Primer](https://github.com/jthack/PIPE)</sub>
