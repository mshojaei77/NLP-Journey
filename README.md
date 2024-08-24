## NLP Journey - Roadmap to Learn LLMs from Scratch with Modern NLP Methods in 2024

### Overview

This repository provides a comprehensive guide on leveraging Jupyter Notebooks for Natural Language Processing (NLP) tasks, culminating in the understanding and application of Large Language Models (LLMs). It focuses on the essential technical skills required for LLM and NLP-related jobs in 2024.

**Note:** The `Resources` column now includes text-based tutorials, papers, and relevant GitHub repositories. 


## Roadmap

| Topic                                                    | Resources | Practices |
|----------------------------------------------------------|-----------|-----------|
| **1. Foundations**                                        |           |           |
|  - Programming Languages: Python, C, Assembly, Rust     |  [] | [] |
|  - Mathematics:                                         |           |           |
|    - Linear Algebra: Matrices, Vectors, Operations     | [3Blue1Brown - Essence of Linear Algebra, Immersive Linear Algebra, Khan Academy - Linear Algebra] | [] |
|    - Calculus: Derivatives, Gradients, Chain Rule      | [Khan Academy - Calculus] | [] |
|    - Probability & Statistics: Distributions, Hypothesis Testing | [StatQuest with Josh Starmer, AP Statistics Intuition by Ms Aerin, Khan Academy - Probability and Statistics] | [] |
|  - Data Structures & Algorithms                        | []        | []        |
| **2. Introduction to NLP & Core Concepts**               |           |           |
|  - Core NLP Concepts: Syntax, Semantics, Pragmatics, Discourse | []        | []        |
|  - Text Preprocessing & Feature Engineering:            |           |           |
|    - Tokenization: Word, Subword (BPE, SentencePiece), minbpe ([minbpe](https://www.youtube.com/watch?v=zduSFxRajkE&t=1157s)) | []        | []        |
|    - Stemming & Lemmatization                         | []        | []        |
|    - Stop Word Removal, Punctuation Handling          | []        | []        |
|    - Bag-of-Words (BoW), TF-IDF, N-grams             | []        | []        |
|  - Word Embeddings: Word2Vec, GloVe, FastText, Contextual Embeddings (ELMo, BERT) | [Jay Alammar - Illustrated Word2Vec] | [] |
| **3. Core NLP Tasks & Algorithms**                      |           |           | 
|  - Text Classification: Naive Bayes, SVM, Logistic Regression, Deep Learning Classifiers | []        | []        |
|  - Text Clustering: K-Means, Hierarchical Clustering, DBSCAN, OPTICS | []        | []        |
|  - Topic Modeling: Latent Dirichlet Allocation (LDA), Non-negative Matrix Factorization (NMF) | []        | []        |
|  - Named Entity Recognition (NER): NLTK, spaCy, Transformers-based NER | []        | []        |
|  - Sentiment Analysis: Lexicon-Based, Machine Learning, Aspect-Based | []        | []        |
|  - Information Retrieval: TF-IDF, BM25, Query Expansion, Vector Search, Semantic Search | []        | []        |
|  - Language Modeling:  N-gram, Neural (RNNs, Transformers) | [Jake Tae - PyTorch RNN from Scratch, colah's blog - Understanding LSTM Networks] | [] |
| **4. Deep Learning for NLP**                             |           |           |
|  - Neural Networks: Basics, Backpropagation, Perceptron  | [3Blue1Brown - But what is a Neural Network?, freeCodeCamp - Deep Learning Crash Course] | [] |
|  - Deep Learning Frameworks: PyTorch, JAX, TensorFlow  | [Fast.ai - Practical Deep Learning, Patrick Loeber - PyTorch Tutorials] | [] |
|  - Recurrent Neural Networks (RNNs): Sequence Modeling, LSTMs, GRUs, Attention | [] | [] |
|  - Convolutional Neural Networks (CNNs) for Text: Classification, Hierarchical CNNs | []        | []        |
|  - Sequence-to-Sequence Models: Attention, Transformers, T5, BART | []        | []        |
| **5. Large Language Models (LLMs) & Transformers**        |           |           |
|  - Transformer Architecture: Attention, Residual Connections, Layer Normalization, RoPE | [The Illustrated Transformer, The Illustrated GPT-2, Visual intro to Transformers, LLM Visualization, nanoGPT, Attention? Attention!] | [] |
|  - LLMs Architectures & Training:                      |           |           |
|    - GPT, BERT, T5, Llama, PaLM                        | [LLMDataHub, Training a causal language model from scratch, TinyLlama, Causal language modeling, Chinchilla's wild implications, BLOOM, OPT-175 Logbook, LLM 360] | [] |
|    - Emerging Architectures: DeepSeek-v2, Jamba, Mixture of Experts (MoE) | [DeepSeek-v2](https://arxiv.org/abs/2405.04434), [Jamba](https://arxiv.org/abs/2403.19887), [Mixture of Experts Explained](https://huggingface.co/blog/moe)  | [] |
|  - Prompt Engineering:                                 |           |           |
|    - Techniques: Zero-Shot, Few-Shot, Chain-of-Thought, ReAct  | [Prompt engineering guide](https://www.promptingguide.ai/)  | [] |
|    - Task-Specific Prompting (e.g., Code Generation)  | [] | [] |
|    - Structuring Outputs: Templates, JSON, LMQL, Outlines, Guidance | [Outlines - Quickstart](https://outlines-dev.github.io/outlines/quickstart/), [LMQL - Overview](https://lmql.ai/docs/language/overview.html) | [] |
|  - Fine-tuning & Adaptation:                             |           |           |
|    - Supervised Fine-Tuning (SFT)                      | [The Novice's LLM Training Guide, Fine-Tune Your Own Llama 2 Model, Padding Large Language Models, A Beginner's Guide to LLM Fine-Tuning] | [] |
|    - Parameter-Efficient Fine-tuning (PEFT): LoRA, Adapters, Prompt Tuning | [LoRA insights](https://lightning.ai/pages/community/lora-insights/) | [] |
|    - Reinforcement Learning from Human Feedback (RLHF): PPO, DPO | [Distilabel, An Introduction to Training LLMs using RLHF, Illustration RLHF, Preference Tuning LLMs, LLM Training: RLHF and Its Alternatives, Fine-tune Mistral-7b with DPO] | [] |
|    - Model Merging:  SLERP, DARE/TIES, FrankenMoEs  | [Merge LLMs with mergekit](https://mlabonne.github.io/blog/posts/2024-01-08_Merge_LLMs_with_mergekit.html), [DARE/TIES](https://arxiv.org/abs/2311.03099), [Phixtral](https://huggingface.co/mlabonne/phixtral-2x2_8) | [] |
| **6.  Multimodal Learning & Applications**                 |           |           |
|  - Multimodal LLMs: CLIP, ViT, LLaVA                    |  [] | [] |
|  - Vision-Language Tasks: Image Captioning, VQA, Visual Reasoning | []        | []        |
|  - Text-to-Image Generation, Video Understanding        | []        | []        |
|  - Emerging Trends: Neuro-Symbolic AI, LLMs for Robotics | []        | []        |
| **7. Retrieval Augmented Generation (RAG)**              |           |           |
|  - Building a Vector Database:                          |           |           |
|    - Document Loaders, Text Splitters                  | [LangChain - Text splitters](https://python.langchain.com/docs/modules/data_connection/document_transformers/), [Sentence Transformers library](https://www.sbert.net/) | [] |
|    - Embedding Models                                    | [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) | [] |
|    - Vector Databases: Chroma, Pinecone, Milvus, FAISS, Annoy | [The Top 5 Vector Databases](https://www.datacamp.com/blog/the-top-5-vector-databases) | [] |
|  - RAG Pipelines & Techniques:                        |           |           |
|    - Orchestrators: LangChain, LlamaIndex, FastRAG    | [Llamaindex - High-level concepts](https://docs.llamaindex.ai/en/stable/getting_started/concepts.html), [Pinecone - Retrieval Augmentation](https://www.pinecone.io/learn/series/langchain/langchain-retrieval-augmentation/) | [] |
|    - Query Expansion, Re-ranking, HyDE                 |  [HyDE](https://arxiv.org/abs/2212.10496) | [] |
|    - RAG Fusion                                        |  [RAG-fusion](https://github.com/Raudaschl/rag-fusion) | [] |
|    - Evaluation: Context Precision/Recall, Faithfulness, Relevancy, Ragas, DeepEval | [LangChain - Q&A with RAG](https://python.langchain.com/docs/use_cases/question_answering/quickstart), [RAG pipeline - Metrics](https://docs.ragas.io/en/stable/concepts/metrics/index.html) | [] |
|  - Advanced RAG:                                        |           |           |
|    - Query Construction: SQL, Cypher                    | [LangChain - Query Construction](https://blog.langchain.dev/query-construction/), [LangChain - SQL](https://python.langchain.com/docs/use_cases/qa_structured/sql) | [] |
|    - Agents & Tools: Google Search, Wikipedia, Python, Jira | [Pinecone - LLM agents](https://www.pinecone.io/learn/series/langchain/langchain-agents/), [LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/) | [] |
|    - Programmatic LLMs: DSPy                          |  [DSPy in 8 Steps](https://dspy-docs.vercel.app/docs/building-blocks/solving_your_task) | [] |
| **8. Deployment & Productionizing LLMs**                  |           |           |
|  - Deployment Strategies:                             |           |           |
|    - Local Servers: LM Studio, Ollama                   | [] | [] |
|    - Cloud Deployment: AWS, GCP, Azure, SkyPilot ([SkyPilot](https://github.com/skypilot-org/skypilot)), Specialized Hardware (TPUs) |  [] | [] |
|    - Serverless Functions, Edge Deployment (MLC LLM)   |  [] | [] |
|  - Inference Optimization:                               |           |           |
|    - Quantization: GPTQ, EXL2, GGUF, llama.cpp         | [Introduction to quantization, Quantize Llama models with llama.cpp, 4-bit LLM Quantization with GPTQ, ExLlamaV2: The Fastest Library to Run LLMs] | [] |
|    - Flash Attention, Key-Value Cache (MQA, GQA)     | [Flash-Attention](https://github.com/Dao-AILab/flash-attention), [Multi-Query Attention](https://arxiv.org/abs/1911.02150), [Grouped-Query Attention](https://arxiv.org/abs/2305.13245) | [] | 
|    - Knowledge Distillation, Pruning                   |  [] | [] |
|    - Speculative Decoding                              | [Assisted Generation](https://huggingface.co/blog/assisted-generation) | [] |
|  - Building LLM Applications:                         |           |           |
|    - APIs: OpenAI, Google, Anthropic, Cohere, OpenRouter, Hugging Face |  [] | [] |
|    - Web Frameworks: Gradio, Streamlit                 | [Streamlit - Build a basic LLM app](https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps) | [] |
|    - User Interfaces, Chatbots                        |  [] | [] |
|  - MLOps for LLMs:                                      |           |           |
|    - CI/CD, Monitoring, Model Management               | []        | []        |
|    - Experiment Tracking, Model Versioning             | []        | []        |
|    - Data & Model Pipelines                            |  [] | [] |
|  - LLM Security:                                         |           |           | 
|    - Prompt Hacking: Injection, Leaking, Jailbreaking  | [OWASP LLM Top 10, Prompt Injection Primer] | [] |
|    - Backdoors: Data Poisoning, Trigger Backdoors     |  [] | [] |
|    - Defensive Measures: Red Teaming, Garak ([garak](https://github.com/leondz/garak/)), Langfuse ([Langfuse](https://github.com/langfuse/langfuse)) | [LLM Security, Red teaming LLMs] | [] | 

