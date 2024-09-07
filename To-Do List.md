## To-Do List: LLM Course

**Module I: Foundations of NLP and Text Data**

**Week 1: Introduction to NLP and Text Data Fundamentals**

- [x] Learn about NLP and its applications.
- [x] Understand text as data: challenges and opportunities.
- [ ] Learn text data fundamentals: characters, words, sentences, documents.
- [ ] Explore text data sources and formats: corpora, web scraping, APIs.
- [ ] **Lab 1:** Explore different text data sources and formats (Project Gutenberg, Wikipedia dumps, Twitter API).
- [ ] **Lab 2:** Implement basic text analysis techniques (word frequency counting, sentence length analysis).

**Week 2: Curating and Cleaning Large Text Datasets & Text Preprocessing**

- [ ] Understand the importance of data quality for LLMs.
- [ ] Learn about dataset utilities: identifying near duplicates, passive voice detection.
- [ ] Learn how to handle special characters and punctuation.
- [ ] Understand text preprocessing: preparing text data for model consumption.
- [ ] **Lab 3:** Clean and preprocess a large text dataset (removing HTML tags, handling special characters).
- [ ] **Lab 4:** Implement a pipeline for detecting and removing near-duplicate text segments.

**Week 3: Tokenization & Text Normalization**

- [ ] Understand tokenization: breaking down text into smaller units (tokens).
- [ ] Learn about word tokenization: segmenting text into individual words.
- [ ] Understand Byte Pair Encoding (BPE): learning subword units from data.
- [ ] Learn about subword tokenization (BPE and others): advantages and limitations.
- [ ] Compare BPE implementations: explore different BPE libraries.
- [ ] Learn about Hugging Face Tokenizers: a powerful library for tokenization.
- [ ] Train a tokenizer with Hugging Face Tokenizers: build custom tokenizers.
- [ ] Understand text normalization: standardizing text representation.
- [ ] Learn about lowercasing: converting text to lowercase.
- [ ] Understand stemming: reducing words to their base form.
- [ ] Compare lemmatization and stemming: different normalization techniques.
- [ ] Learn about stop word removal: filtering out common words.
- [ ] Understand punctuation handling: removing or standardizing punctuation.
- [ ] **Lab 5:** Implement different tokenization techniques (word tokenization, BPE) using NLTK and Hugging Face Tokenizers.
- [ ] **Lab 6:** Train a custom tokenizer using Hugging Face Tokenizers.
- [ ] **Lab 7:** Implement and compare different text normalization techniques (stemming, lemmatization).

**Week 4: Feature Engineering & Text Augmentation**

- [ ] Understand feature engineering: representing text data numerically.
- [ ] Learn about Bag-of-Words (BoW): creating feature vectors based on word counts.
- [ ] Understand TF-IDF: weighing words based on their importance in a document and corpus.
- [ ] Learn about N-grams: capturing sequences of words as features.
- [ ] Understand N-gram language models: predicting the next word based on previous words.
- [ ] Learn about bigram language model (language modeling): a practical example.
- [ ] Understand text augmentation: increasing the size and diversity of the training data.
- [ ] **Lab 8:** Implement BoW and TF-IDF feature extraction using scikit-learn.
- [ ] **Lab 9:** Build an N-gram language model using Python.
- [ ] **Lab 10:** Implement different text augmentation techniques (synonym replacement, random insertion).

**Module II: Word Embeddings and Language Modeling**

**Week 5: Word Embeddings**

- [ ] Understand word embeddings: representing words as dense vectors.
- [ ] Learn about Word2Vec: learning word embeddings using a shallow neural network.
- [ ] Understand GloVe: learning word embeddings based on co-occurrence statistics.
- [ ] Learn about FastText: learning word embeddings that capture subword information.
- [ ] Understand contextual embeddings: representing words based on their context.
- [ ] Learn about ELMo: learning contextualized word embeddings using bidirectional LSTMs.
- [ ] Understand embedding layers vs. linear layers: their role in neural networks.
- [ ] Learn about fine-tuning word embeddings for LLMs: adapting pre-trained embeddings for specific tasks.
- [ ] **Lab 11:** Train and visualize word embeddings using Word2Vec, GloVe, and FastText.
- [ ] **Lab 12:** Explore the properties of pre-trained word embeddings (word similarity, analogy tasks).


**Week 6: Language Modeling Basics & Neural Language Models**

- [ ] Understand language modeling basics: predicting the next word.
- [ ] Learn about neural language models: using neural networks for language modeling.
- [ ] Understand masked language modeling (MLM): predicting masked words in a sentence (BERT-style).
- [ ] Understand causal language modeling (CLM) objective: training autoregressive language models (GPT-style).
- [ ] **Lab 13:** Implement a simple neural language model using an RNN in PyTorch.
- [ ] **Lab 14:** Implement a masked language model training loop in PyTorch.


**Module III: Transformer Architectures and their Components**


**Week 7: Transformer Architecture Overview & Attention Mechanisms**

- [ ] Understand transformer architecture overview: for sequence-to-sequence tasks.
- [ ] Learn about the transformer architecture: deep dive into encoder and decoder components.
- [ ] Understand attention mechanisms: the core innovation behind transformers.
- [ ] Learn about self-attention and multi-head attention: how attention works.
- [ ] Understand scaled dot-product attention: implementation details.
- [ ] Learn about coding attention mechanisms: implementing attention from scratch in PyTorch.
- [ ] Compare efficient multi-head attention implementations: explore different optimizations.
- [ ] **Lab 15:** Implement a basic self-attention mechanism in PyTorch.
- [ ] **Lab 16:** Implement multi-head attention and compare its performance with single-head attention.


**Week 8: Sparse Attention and Long-Context Transformers & Core Transformer Components**

- [ ] Understand sparse attention and long-context transformers: addressing limitations of standard attention.
- [ ] Learn about core transformer components:
    - [ ] Encoder and decoder stacks: building deep transformer networks.
    - [ ] Positional encoding: injecting positional information into the model.
    - [ ] Feed-forward networks: adding non-linearity to the model.
    - [ ] Residual connections: enabling the training of very deep networks.
    - [ ] Layer normalization and alternatives (RMSNorm): stabilizing training and improving performance.
- [ ] **Lab 17:** Implement a transformer encoder with positional encoding and feed-forward networks.
- [ ] **Lab 18:** Explore different layer normalization techniques and their impact on model performance.


**Module IV: Understanding Key LLM Architectures**


**Week 9: Generative Pre-trained Transformer Models (GPT) & BERT and its Variants**

- [ ] Understand generative pre-trained transformer models (GPT): powerful autoregressive language models.
- [ ] Learn about building the GPT architecture: understanding the layers and components.
- [ ] Learn about the GPT family (GPT-1, GPT-2, GPT-3, GPT-4): evolution of the GPT architecture.
- [ ] Understand BERT and its variants: a powerful bidirectional language model.
- [ ] Learn about BERT-base, BERT-large, RoBERTa: exploring different BERT configurations.
- [ ] Revisit masked language models (BERT-style): the pre-training objective.
- [ ] **Lab 19:** Fine-tune a pre-trained GPT model for text generation using Hugging Face Transformers.
- [ ] **Lab 20:** Fine-tune a pre-trained BERT model for text classification using Hugging Face Transformers.


**Week 10: T5, BART, and Other Notable Architectures**

- [ ] Understand T5 (Text-to-Text Transfer Transformer): a unified framework for NLP tasks.
- [ ] Explore T5 and its applications: various use cases of T5.
- [ ] Understand BART (Bidirectional and Auto-Regressive Transformers): combining encoder-decoder and autoregressive approaches.
- [ ] Learn about other notable architectures: Llama, RoPE, RMSNorm, GQA, MoE.
- [ ] Understand retrieval-augmented LLMs (RAG, RETRO, etc.): enhancing LLMs with external knowledge.
- [ ] **Lab 21:** Fine-tune a pre-trained T5 model for a specific task (summarization, translation).
- [ ] **Lab 22:** Explore and experiment with different pre-trained models from Hugging Face Model Hub.


**Module V: Pre-training Large Language Models**


**Week 11: The Importance of Pre-training & Pre-training Objectives**

- [ ] Understand the importance of pretraining: building a general-purpose language understanding foundation.
- [ ] Learn about pre-training objectives: language model training, masked language modeling (MLM), causal language modeling (CLM).
- [ ] **Lab 23:** Implement a simple language model training loop in PyTorch.


**Week 12: Pre-training Process & Evaluating Pretraining Performance**

- [ ] Understand the pre-training process: training loop, optimization, data handling.
- [ ] Learn about training loop and optimization: implementing the pre-training process.
- [ ] Understand evaluating pretraining performance: metrics for language model evaluation (perplexity).
- [ ] Learn about pretraining GPT on Project Gutenberg: a practical example using a smaller dataset.
- [ ] **Lab 24:** Pre-train a small language model on a text dataset.
- [ ] **Lab 25:** Evaluate the performance of a pre-trained language model using perplexity.


**Week 13: Optimizing Pre-training & Ethical Considerations**

- [ ] Learn about optimizing hyperparameters for pre-training: finding the best settings for training (learning rate, batch size).
- [ ] Understand FLOPS analysis: measuring the computational cost of LLMs.
- [ ] Understand ethical challenges in pretraining data (bias, privacy): mitigating potential issues.
- [ ] **Lab 26:** Experiment with different hyperparameters to optimize pre-training performance.
- [ ] **Discussion:** Ethical implications of using large datasets for pre-training LLMs.


**Module VI: Fine-tuning LLMs for Downstream Tasks**


**Week 14: Fine-tuning Basics & Fine-tuning for Text Classification**

- [ ] Understand fine-tuning basics: adapting a pre-trained LLM for specific tasks.
- [ ] Learn about fine-tuning for text classification: a common downstream task.
- [ ] Learn about adapting GPT for classification: modifying the architecture for classification.
- [ ] Understand fine-tuning on labeled data: using supervised learning for fine-tuning.
- [ ] Learn about evaluation metrics for classification: accuracy, precision, recall, F1-score.
- [ ] **Lab 27:** Fine-tune a pre-trained LLM for text classification using a labeled dataset.
- [ ] **Lab 28:** Evaluate the performance of a fine-tuned LLM for text classification using different metrics.


**Week 15: Fine-tuning Different Models and Handling Data Imbalance**

- [ ] Learn about fine-tuning different models on IMDB movie reviews: a practical example.
- [ ] Revisit dataset utilities (near duplicates, passive voice): techniques for data quality.
- [ ] Understand handling data imbalance in fine-tuning: addressing bias in the training data.
- [ ] **Lab 29:** Compare the performance of different pre-trained LLMs fine-tuned for the same task.
- [ ] **Lab 30:** Implement techniques for handling data imbalance during fine-tuning (oversampling, undersampling).


**Week 16: Instruction Tuning and Alignment**

- [ ] Understand instruction tuning and alignment: training LLMs to follow instructions and align with human values.
- [ ] Learn about supervised fine-tuning with instruction datasets: creating and using instruction datasets.
- [ ] Learn about evaluating instruction responses (OpenAI API, Ollama): assessing the quality of LLM outputs.
- [ ] Learn about generating a dataset for instruction fine-tuning: building custom instruction datasets.
- [ ] Understand challenges in multi-turn instruction fine-tuning: handling complex conversations and context.
- [ ] **Lab 31:** Fine-tune an LLM using an instruction dataset.
- [ ] **Lab 32:** Evaluate the performance of an instruction-tuned LLM using different metrics.



**Module VII: Parameter-Efficient Fine-Tuning (PEFT) and RLHF**


**Week 17: Understanding LoRA & Implementing LoRA for Fine-tuning**

- [ ] Understand LoRA (Low-Rank Adaptation): a technique for efficient fine-tuning.
- [ ] Learn about implementing LoRA for fine-tuning: practical examples in PyTorch.
- [ ] Understand the benefits of using LoRA: reduced memory footprint and faster training times.
- [ ] **Lab 33:** Implement LoRA for fine-tuning a pre-trained LLM.
- [ ] **Lab 34:** Compare the performance and efficiency of fine-tuning with and without LoRA.


**Week 18: Alternatives to LoRA & Reinforcement Learning from Human Feedback (RLHF)**

- [ ] Learn about alternatives to LoRA (Adapters, Prefix Tuning): exploring other PEFT techniques.
- [ ] Learn about generating a preference dataset (Llama 3.1 70B, Ollama): collecting human preferences for LLM outputs.
- [ ] Understand direct preference optimization (DPO) for LLM alignment: optimizing LLM behavior based on human preferences.
- [ ] Learn about training LLMs to align with human values: ethical considerations and best practices.
- [ ] **Lab 35:** Implement and compare different PEFT techniques (LoRA, Adapters).
- [ ] **Lab 36:** Build a simple preference dataset for LLM evaluation.


**Module VIII: Optimizing LLM Training**


**Week 19: Initialization and Optimization & Learning Rate Schedulers**

- [ ] Understand initialization and optimization (AdamW): choosing appropriate initialization and optimization strategies.
- [ ] Learn about learning rate schedulers: adapting the learning rate during training for improved convergence.
- [ ] **Lab 37:** Experiment with different initialization techniques and optimizers for LLM training.
- [ ] **Lab 38:** Implement and compare different learning rate schedulers (cosine annealing, step decay).


**Week 20: Gradient Clipping and Accumulation & Mixed Precision Training**

- [ ] Understand gradient clipping and accumulation: handling exploding gradients and enabling larger batch sizes.
- [ ] Learn about mixed precision training: using lower precision floats (fp16, bf16, fp8) for faster training.
- [ ] **Lab 39:** Implement gradient clipping and accumulation in PyTorch.
- [ ] **Lab 40:** Implement mixed precision training using PyTorch's automatic mixed precision (AMP) feature.


**Week 21: Distributed Training & Large Batch Training and its Challenges**

- [ ] Understand distributed training: training LLMs across multiple GPUs or machines using techniques like DDP and ZeRO.
- [ ] Understand large batch training and its challenges: understanding the trade-offs and optimization issues.
- [ ] **Lab 41:** Implement distributed training using PyTorch DDP.
- [ ] **Lab 42:** Experiment with different batch sizes and observe their impact on training time and performance.


**Week 22: Device Optimization**

- [ ] Understand device optimization (CPU, GPU): choosing the right hardware for LLM training and inference.
- [ ] **Lab 43:** Compare the performance of LLM training on CPU and GPU.


**Module IX: LLM Inference and Optimization**


**Week 23: KV-Cache & Quantization**

- [ ] Understand KV-Cache: caching key-value pairs for faster inference.
- [ ] Understand quantization: reducing model size and memory footprint by using lower precision representations.
- [ ] **Lab 44:** Implement KV-Cache for faster LLM inference.
- [ ] **Lab 45:** Implement model quantization using different techniques (post-training quantization, quantization-aware training).


**Week 24: Pruning and Distillation & Other Inference Optimization Techniques**

- [ ] Understand pruning and distillation for lightweight models: reducing model size and complexity.
- [ ] Learn about Flash Attention: an efficient attention mechanism for faster inference.
- [ ] Understand knowledge distillation: training a smaller model to mimic the behavior of a larger model.
- [ ] Understand pruning: removing less important weights from the model.
- [ ] Understand sparse models: utilizing sparsity for efficient computation and memory usage.
- [ ] Learn about model compression techniques: overview of various techniques for reducing model size and complexity.
- [ ] Learn about speculative decoding: generating multiple output candidates in parallel for faster inference.
- [ ] **Lab 46:** Implement knowledge distillation to train a smaller LLM.
- [ ] **Lab 47:** Implement model pruning using different techniques (magnitude-based pruning, movement pruning).


**Module X: Deploying LLMs**


**Week 25: Deployment Options & API Development and Web App Deployment**

- [ ] Understand deployment options: local servers, cloud deployment (AWS, GCP, Azure), serverless functions, edge deployment.
- [ ] Learn about API development and web app deployment: building APIs and web applications for accessing LLMs.
- [ ] **Lab 48:** Deploy an LLM on a local server using Flask or FastAPI.
- [ ] **Lab 49:** Deploy an LLM as a serverless function using AWS Lambda or Google Cloud Functions.


**Week 26: Modifying Context Windows and Monitoring LLMs**

- [ ] Understand modifying context windows for long inputs: handling long sequences during inference.
- [ ] Learn about monitoring and managing LLMs in production: tracking performance, identifying issues, and ensuring reliability.
- [ ] Understand handling model drift: adapting to changes in data distribution over time.
- [ ] Learn about post-deployment model tuning: continuously improving model performance after deployment.
- [ ] **Lab 50:** Implement a system for monitoring LLM performance in a production environment.
- [ ] **Lab 51:** Implement techniques for handling model drift (retraining on new data, online learning).


**Week 27: LLM Security & Defensive Measures**

- [ ] Understand LLM security: mitigating potential security risks associated with LLMs.
- [ ] Learn about prompt hacking and backdoors: manipulating inputs to elicit unwanted behaviors.
- [ ] Learn about prompt injection: injecting malicious instructions into prompts.
- [ ] Learn about prompt leaking: extracting sensitive information through carefully crafted prompts.
- [ ] Learn about jailbreaking: bypassing safety restrictions imposed on LLMs.
- [ ] Learn about backdoors: inserting hidden functionalities into LLMs that can be triggered by specific inputs.
- [ ] Learn about data poisoning: injecting malicious data into training datasets to compromise model behavior.
- [ ] Learn about trigger backdoors: backdoors that are activated by specific trigger phrases or patterns.
- [ ] Learn about defensive measures: strategies and best practices for protecting LLMs from attacks.
- [ ] Learn about red teaming: simulating attacks to identify vulnerabilities.
- [ ] Learn about input sanitization: validating and cleaning user inputs to prevent malicious prompts.
- [ ] Learn about output monitoring: analyzing LLM outputs to detect suspicious or harmful content.
- [ ] **Discussion:** Ethical and societal implications of LLM security vulnerabilities.
- [ ] **Lab 52:** Implement input sanitization techniques to prevent prompt injection attacks.
- [ ] **Lab 53:** Implement output monitoring techniques to detect potentially harmful LLM outputs.


**Module XI: LLM Applications**


**Week 28: Text-based Applications: Chatbots, Code Generation, and Summarization**

- [ ] Learn about chatbot development: building conversational agents using LLMs.
- [ ] Learn about code generation assistants: using LLMs to generate code and automate software development.
- [ ] Learn about summarization engines: extracting key information from text and generating concise summaries.
- [ ] **Lab 54:** Build a simple chatbot using a pre-trained LLM and a web framework like Gradio or Streamlit.
- [ ] **Lab 55:** Implement a code generation assistant that can generate code snippets based on natural language descriptions.
- [ ] **Lab 56:** Build a text summarization engine using a pre-trained LLM.


**Week 29: Text-based Applications: Sentiment Analysis, NER, and Topic Modeling**

- [ ] Learn about Sentiment Analysis: 
    - [ ] Lexicon-Based Approach: Using sentiment lexicons to analyze the emotional tone of text.
    - [ ] Machine Learning Approach: Training classifiers to predict sentiment.
    - [ ] Aspect-Based Sentiment Analysis: Identifying sentiment towards specific aspects of a product or service.
- [ ] Learn about Named Entity Recognition (NER): 
    - [ ] Rule-Based Systems: Using handcrafted rules to identify named entities.
    - [ ] Machine Learning-Based Systems: Training models to recognize named entities.
    - [ ] NLTK and spaCy: Exploring popular NER libraries.
- [ ] Learn about Topic Modeling: 
    - [ ] Latent Dirichlet Allocation (LDA): A probabilistic model for discovering latent topics in a collection of documents.
    - [ ] Non-Negative Matrix Factorization (NMF): A matrix decomposition technique for topic modeling.
- [ ] **Lab 57:** Implement a sentiment analysis system using a lexicon-based approach and a machine learning approach.
- [ ] **Lab 58:** Implement a named entity recognition system using NLTK or spaCy.
- [ ] **Lab 59:** Implement a topic modeling system using LDA or NMF.


**Week 30: Multimodal Learning and Applications**

- [ ] Understand multimodal LLMs: LLMs that can process and generate both text and other modalities (images, audio, video).
- [ ] Learn about handling images, audio, and video: preprocessing and encoding multimodal data for LLM consumption.
- [ ] Learn about VQVAE and VQGAN: techniques for learning discrete representations of images.
- [ ] Learn about diffusion transformer: a powerful architecture for text-to-image generation.
- [ ] Learn about combining text and code (Code-LLMs, Codex, GPT-Engineer): LLMs that can understand and generate code.
- [ ] Learn about self-training and semi-supervised learning: leveraging unlabeled data to improve LLM performance.
- [ ] Learn about multimodal applications:
    - [ ] Text-to-Image Generation: Generating images from textual descriptions.
    - [ ] Video Understanding: Analyzing and generating descriptions of videos.
- [ ] **Lab 60:** Implement a text-to-image generation system using a pre-trained diffusion model.
- [ ] **Lab 61:** Implement a video understanding system that can generate captions or summaries for videos.


**Module XII: Advanced Topics and Future Directions**


**Week 31: Neuro-Symbolic AI & LLMs for Robotics**

- [ ] Understand neuro-symbolic AI: combining neural networks with symbolic reasoning for more robust and interpretable AI systems.
- [ ] Learn about LLMs for robotics: using LLMs to control robots, generate robot behavior, and enable human-robot interaction through natural language.
- [ ] **Discussion:** Potential applications and challenges of neuro-symbolic AI and LLMs in robotics.


**Week 32: MLOps for LLMs**

- [ ] Understand MLOps for LLMs: applying MLOps principles to the development, deployment, and management of LLMs.
- [ ] Learn about CI/CD for LLM workflows: automating the building, testing, and deployment of LLMs.
- [ ] Learn about monitoring: tracking LLM performance and identifying issues in production.
- [ ] Learn about model management: versioning, storing, and deploying different LLM models.
- [ ] Learn about experiment tracking: managing and comparing LLM experiments.
- [ ] Learn about data and model pipelines: building efficient data processing and model training pipelines.
- [ ] **Lab 62:** Implement a CI/CD pipeline for automating the deployment of an LLM.
- [ ] **Lab 63:** Implement a system for tracking and managing LLM experiments.


**Week 33: Capstone Project & Future Directions**

- [ ] **Capstone Project:** Build and deploy a real-world LLM application that addresses a specific problem or task.
- [ ] Explore future directions: emerging trends and research directions in the field of LLMs, such as:
    - [ ] More efficient and scalable LLM architectures.
    - [ ] Improved methods for LLM alignment and safety.
    - [ ] New applications of LLMs in various domains.
    - [ ] Ethical and societal implications of LLMs.
- [ ] **Project Presentations:** Prepare and present capstone project, discussing findings and insights.
- [ ] **Final Discussion:** Reflect on the course content and discuss the future of LLMs.


**Assessment**

- [ ] Complete weekly quizzes to assess understanding of core concepts.
- [ ] Complete programming assignments to develop practical skills in building, training, fine-tuning, and deploying LLMs.
- [ ] Complete the capstone project involving the development of a substantial LLM application.
- [ ] Actively participate in class discussions and online forums.


