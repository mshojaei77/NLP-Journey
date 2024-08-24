## NLP Journey - Roadmap to Learn LLMs from Scratch with Modern NLP Methods in 2024

### Overview

This repository provides a comprehensive guide on leveraging Jupyter Notebooks for Natural Language Processing (NLP) tasks, culminating in the understanding and application of Large Language Models (LLMs). It focuses on the essential technical skills required for LLM and NLP-related jobs in 2024.

**Note:** The `Resources` column now includes text-based tutorials, papers, and relevant GitHub repositories. The `Practices` column now includes Jupyter Notebooks where available.

## 1. Introduction to NLP & Core Concepts

| Topic                                                    | Resources | Practices |
|----------------------------------------------------------|-----------|-----------|
| - Core NLP Concepts: Syntax, Semantics, Pragmatics, Discourse | [Speech and Language Processing (Jurafsky & Martin)](https://web.stanford.edu/~jurafsky/slp3/) | [] |
| - Text Preprocessing & Feature Engineering:            |           |           |
|   - Tokenization: Word, Subword (BPE, SentencePiece) |  [minbpe Tutorial](https://www.youtube.com/watch?v=zduSFxRajkE&t=1157s), [Hugging Face Tokenizers](https://huggingface.co/docs/tokenizers/index) | []        |
|   - Stemming & Lemmatization                         | [NLTK Stemming and Lemmatization](https://www.nltk.org/howto/stem.html) | []        |
|   - Stop Word Removal, Punctuation Handling          | [NLTK Stop Words](https://www.nltk.org/book/ch02.html#stop-words-corpus) | []        |
|   - Bag-of-Words (BoW), TF-IDF, N-grams             | [Scikit-learn: Text Feature Extraction](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) | []        |
| - Word Embeddings: Word2Vec, GloVe, FastText, Contextual Embeddings (ELMo, BERT) | [Jay Alammar - Illustrated Word2Vec](https://jalammar.github.io/illustrated-word2vec/), [Gensim Word2Vec](https://radimrehurek.com/gensim/models/word2vec.html), [Stanford GloVe](https://nlp.stanford.edu/projects/glove/) | [] |

## 2. Core NLP Tasks & Algorithms

| Topic                                                    | Resources | Practices |
|----------------------------------------------------------|-----------|-----------|
| - Text Classification: Naive Bayes, SVM, Logistic Regression, Deep Learning Classifiers | [Scikit-learn Text Classification](https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html), [Hugging Face Text Classification](https://huggingface.co/docs/transformers/tasks/sequence_classification) |  [] |
| - Text Clustering: K-Means, Hierarchical Clustering, DBSCAN, OPTICS | [Scikit-learn Clustering](https://scikit-learn.org/stable/modules/clustering.html) |  [] |
| - Topic Modeling: Latent Dirichlet Allocation (LDA), Non-negative Matrix Factorization (NMF) | [Gensim Topic Modeling](https://radimrehurek.com/gensim/auto_examples/tutorials/run_lda.html), [Scikit-learn NMF](https://scikit-learn.org/stable/modules/decomposition.html#non-negative-matrix-factorization-nmf-or-nnmf) |  [] |
| - Named Entity Recognition (NER): NLTK, spaCy, Transformers-based NER | [NLTK NER](https://www.nltk.org/book/ch07.html), [spaCy NER](https://spacy.io/usage/linguistic-features#named-entities), [Hugging Face NER](https://huggingface.co/docs/transformers/tasks/token-classification) |  [] |
| - Sentiment Analysis: Lexicon-Based, Machine Learning, Aspect-Based | [NLTK Sentiment Analysis](https://www.nltk.org/howto/sentiment.html), [TextBlob Sentiment Analysis](https://textblob.readthedocs.io/en/dev/quickstart.html#sentiment-analysis), [VADER Sentiment Analysis](https://github.com/cjhutto/vaderSentiment) |  [] |
| - Information Retrieval: TF-IDF, BM25, Query Expansion, Vector Search, Semantic Search | [Elasticsearch](https://www.elastic.co/), [Solr](https://lucene.apache.org/solr/), [Pinecone](https://www.pinecone.io/) |  [] |
| - Language Modeling:  N-gram, Neural (RNNs, Transformers) | [Stanford NLP: N-gram Language Models](https://nlp.stanford.edu/fsnlp/lm.html), [Hugging Face Language Modeling](https://huggingface.co/docs/transformers/tasks/language_modeling) |  [] |

## 3. Deep Learning for NLP 

| Topic                                                    | Resources | Practices |
|----------------------------------------------------------|-----------|-----------|
| - Neural Networks: Basics, Backpropagation, Perceptron  | [3Blue1Brown - Neural Networks](https://www.youtube.com/watch?v=aircAruvnKk), [freeCodeCamp - Deep Learning Crash Course](https://www.youtube.com/watch?v=VyWAvY2CF9c) |  [] |
| - Deep Learning Frameworks: PyTorch, JAX, TensorFlow  | [PyTorch Tutorials](https://pytorch.org/tutorials/), [JAX Documentation](https://jax.readthedocs.io/en/latest/), [TensorFlow Tutorials](https://www.tensorflow.org/tutorials) |  [] |
| - Recurrent Neural Networks (RNNs): Sequence Modeling, LSTMs, GRUs, Attention | [colah's blog: Understanding LSTMs](https://colah.github.io/posts/2015-08-Understanding-LSTMs/), [Andrej Karpathy: The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) |  [] |
| - Convolutional Neural Networks (CNNs) for Text: Classification, Hierarchical CNNs | [Understanding Convolutional Neural Networks for NLP](http://www.wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp/), [Kim Yoon: Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882) |  [] |
| - Sequence-to-Sequence Models: Attention, Transformers, T5, BART | [Jay Alammar: The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/), [Google AI Blog: Transformer Networks](https://ai.googleblog.com/2017/08/transformer-networks-state-of-art.html), [Hugging Face: T5](https://huggingface.co/docs/transformers/model_doc/t5), [Hugging Face: BART](https://huggingface.co/docs/transformers/model_doc/bart) |  [] |

## 4. Large Language Models (LLMs) & Transformers

| Topic                                                    | Resources | Practices |
|----------------------------------------------------------|-----------|-----------|
| - Transformer Architecture: Attention, Residual Connections, Layer Normalization, RoPE | [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/), [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/), [Visual Intro to Transformers](https://www.youtube.com/watch?v=wjZofJX0v4M&t=187s), [LLM Visualization](https://bbycroft.net/llm), [nanoGPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) |  [] |
| - LLMs Architectures & Training:                      |           |           |
|   - GPT, BERT, T5, Llama, PaLM                        | [LLMDataHub](https://github.com/Zjh-819/LLMDataHub), [Hugging Face: Causal Language Modeling](https://huggingface.co/learn/nlp-course/chapter7/6?fw=pt), [TinyLlama](https://github.com/jzhang38/TinyLlama), [Chinchilla's Wild Implications](https://www.lesswrong.com/posts/6Fpvch8RR29qLEWNH/chinchilla-s-wild-implications), [BLOOM](https://bigscience.notion.site/BLOOM-BigScience-176B-Model-ad073ca07cdf479398d5f95d88e218c4), [OPT-175 Logbook](https://github.com/facebookresearch/metaseq/blob/main/projects/OPT/chronicles/OPT175B_Logbook.pdf), [LLM 360](https://www.llm360.ai/) |  [] |
|   - Emerging Architectures: DeepSeek-v2, Jamba, Mixture of Experts (MoE) | [DeepSeek-v2](https://arxiv.org/abs/2405.04434), [Jamba](https://arxiv.org/abs/2403.19887), [Hugging Face: Mixture of Experts Explained](https://huggingface.co/blog/moe), [Create MoEs with MergeKit Notebook](https://colab.research.google.com/drive/1obulZ1ROXHjYLn6PPZJwRR6GzgQogxxb?usp=sharing) |  [] |
| - Prompt Engineering:                                 |           |           |
|   - Techniques: Zero-Shot, Few-Shot, Chain-of-Thought, ReAct  | [Prompt Engineering Guide](https://www.promptingguide.ai/), [Lilian Weng: Prompt Engineering](https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/), [LLM Prompt Engineering Simplified Book](https://llmnanban.akmmusai.pro/Book/LLM-Prompt-Engineering-Simplified-Book/)  |  [] |
|   - Task-Specific Prompting (e.g., Code Generation)  | [Code Generation with AlphaCodium: From Prompt Engineering to Flow Engineering](https://arxiv.org/abs/2401.08500) |  [] |
|   - Structuring Outputs: Templates, JSON, LMQL, Outlines, Guidance | [Chat Template](https://huggingface.co/blog/chat-templates), [Outlines - Quickstart](https://outlines-dev.github.io/outlines/quickstart/), [LMQL - Overview](https://lmql.ai/docs/language/overview.html), [Microsoft Guidance](https://github.com/microsoft/guidance) |  [] |
| - Fine-tuning & Adaptation:                             |           |           |
|   - Supervised Fine-Tuning (SFT)                      | [Fine-Tune Your Own Llama 2 Model](https://mlabonne.github.io/blog/posts/Fine_Tune_Your_Own_Llama_2_Model_in_a_Colab_Notebook.html), [Padding Large Language Models](https://towardsdatascience.com/padding-large-language-models-examples-with-llama-2-199fb10df8ff), [A Beginner's Guide to LLM Fine-Tuning](https://mlabonne.github.io/blog/posts/A_Beginners_Guide_to_LLM_Finetuning.html), [unslothai](https://github.com/unslothai/unsloth), [Fine-tune Llama 2 with QLoRA Notebook](https://colab.research.google.com/drive/1PEQyJO1-f6j0S_XJ8DV50NkpzasXkrzd?usp=sharing), [Fine-tune CodeLlama using Axolotl Notebook](https://colab.research.google.com/drive/1Xu0BrCB7IShwSWKVcfAfhehwjDrDMH5m?usp=sharing), [Fine-tune Mistral-7b with QLoRA Notebook](https://colab.research.google.com/drive/1o_w0KastmEJNVwT5GoqMCciH-18ca5WS?usp=sharing), [Fine-tune Mistral-7b with DPO Notebook](https://colab.research.google.com/drive/15iFBr1xWgztXvhrj5I9fBv20c7CFOPBE?usp=sharing), [Fine-tune Llama 3 with ORPO Notebook](https://colab.research.google.com/drive/1eHNWg9gnaXErdAa8_mcvjMupbSS6rDvi), [Fine-tune Llama 3.1 with Unsloth Notebook](https://colab.research.google.com/drive/164cg_O7SV7G8kZr_JXqLd6VC7pd86-1Z?usp=sharing) | [] |
|   - Parameter-Efficient Fine-tuning (PEFT): LoRA, Adapters, Prompt Tuning | [LoRA Insights](https://lightning.ai/pages/community/lora-insights/), [Hugging Face: Parameter-Efficient Fine-Tuning](https://huggingface.co/blog/peft) |  [] |
|   - Reinforcement Learning from Human Feedback (RLHF): PPO, DPO |  [Distilabel](https://github.com/argilla-io/distilabel), [An Introduction to Training LLMs using RLHF](https://wandb.ai/ayush-thakur/Intro-RLAIF/reports/An-Introduction-to-Training-LLMs-Using-Reinforcement-Learning-From-Human-Feedback-RLHF---VmlldzozMzYyNjcy), [Hugging Face: Illustration RLHF](https://huggingface.co/blog/rlhf), [Hugging Face: Preference Tuning LLMs](https://huggingface.co/blog/pref-tuning), [LLM Training: RLHF and Its Alternatives](https://magazine.sebastianraschka.com/p/llm-training-rlhf-and-its-alternatives), [Fine-tune Mistral-7b with DPO](https://mlabonne.github.io/blog/posts/Fine_tune_Mistral_7b_with_DPO.html) |  [] |
|   - Model Merging:  SLERP, DARE/TIES, FrankenMoEs  | [Merge LLMs with mergekit](https://mlabonne.github.io/blog/posts/2024-01-08_Merge_LLMs_with_mergekit.html), [DARE/TIES](https://arxiv.org/abs/2311.03099), [Phixtral](https://huggingface.co/mlabonne/phixtral-2x2_8), [MergeKit](https://github.com/cg123/mergekit), [Merge LLMs with MergeKit Notebook](https://colab.research.google.com/drive/1_JS7JKJAQozD48-LhYdegcuuZ2ddgXfr?usp=sharing), [LazyMergekit Notebook](https://colab.research.google.com/drive/1obulZ1ROXHjYLn6PPZJwRR6GzgQogxxb?usp=sharing) |  [] |

## 5.  Multimodal Learning & Applications

| Topic                                                    | Resources | Practices |
|----------------------------------------------------------|-----------|-----------|
| - Multimodal LLMs: CLIP, ViT, LLaVA                    | [OpenAI CLIP](https://openai.com/research/clip), [Google AI Blog: ViT](https://ai.googleblog.com/2020/10/an-image-is-worth-16x16-words.html), [LLaVA](https://llava-vl.github.io/), [MiniCPM-V 2.6](https://github.com/OpenBMB/MiniCPM-V) |  [] |
| - Vision-Language Tasks: Image Captioning, VQA, Visual Reasoning | [Hugging Face: Vision-Language Tasks](https://huggingface.co/docs/transformers/tasks/vision-language-modeling), [Microsoft Kosmos-1](https://arxiv.org/abs/2302.14045), [Google PaLM-E](https://palm-e.github.io/) |  [] |
| - Text-to-Image Generation, Video Understanding        | [Stability AI: Stable Diffusion](https://stability.ai/stable-image), [OpenAI DALL-E 2](https://openai.com/dall-e-2), [Hugging Face: Video Understanding](https://huggingface.co/docs/transformers/tasks/video-classification), [Deep-Live-Cam](https://github.com/hacksider/Deep-Live-Cam) |  [] |
| - Emerging Trends: Neuro-Symbolic AI, LLMs for Robotics | []        | []        |

## 6. Retrieval Augmented Generation (RAG)

| Topic                                                    | Resources | Practices |
|----------------------------------------------------------|-----------|-----------|
| - Building a Vector Database:                          |           |           |
|   - Document Loaders, Text Splitters                  | [LangChain Text Splitters](https://python.langchain.com/docs/modules/data_connection/document_transformers/), [LlamaIndex Data Connectors](https://gpt-index.readthedocs.io/en/latest/guides/primer/data_connectors.html)  |  [] |
|   - Embedding Models                                    | [Sentence Transformers Library](https://www.sbert.net/), [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) |  [] |
|   - Vector Databases: Chroma, Pinecone, Milvus, FAISS, Annoy |  [Chroma](https://www.trychroma.com/), [Pinecone](https://www.pinecone.io/), [Milvus](https://milvus.io/), [FAISS](https://faiss.ai/), [Annoy](https://github.com/spotify/annoy) |  [] |
| - RAG Pipelines & Techniques:                        |           |           |
|   - Orchestrators: LangChain, LlamaIndex, FastRAG    | [LangChain](https://python.langchain.com/), [LlamaIndex](https://docs.llamaindex.ai/en/stable/), [FastRAG](https://github.com/IntelLabs/fastRAG)  |  [] |
|   - Query Expansion, Re-ranking, HyDE                 | [HyDE](https://arxiv.org/abs/2212.10496), [LangChain Retrievers](https://python.langchain.com/docs/modules/data_connection/retrievers/) |  [] |
|   - RAG Fusion                                        | [RAG-fusion](https://github.com/Raudaschl/rag-fusion) |  [] |
|   - Evaluation: Context Precision/Recall, Faithfulness, Relevancy, Ragas, DeepEval |  [Ragas](https://github.com/explodinggradients/ragas/tree/main), [DeepEval](https://github.com/confident-ai/deepeval) |  [] |
| - Advanced RAG:                                        |           |           |
|   - Query Construction: SQL, Cypher                    | [LangChain Query Construction](https://blog.langchain.dev/query-construction/), [LangChain SQL](https://python.langchain.com/docs/use_cases/qa_structured/sql)  |  [] |
|   - Agents & Tools: Google Search, Wikipedia, Python, Jira | [LangChain Agents](https://python.langchain.com/docs/modules/agents/) |  [] |
|   - Programmatic LLMs: DSPy                          | [DSPy](https://github.com/stanfordnlp/dspy)  |  [] |

## 7. Deployment & Productionizing LLMs

| Topic                                                    | Resources | Practices |
|----------------------------------------------------------|-----------|-----------|
| - Deployment Strategies:                             |           |           |
|   - Local Servers: LM Studio, Ollama, Oobabooga, Kobold.cpp                   | [LM Studio](https://lmstudio.ai/), [Ollama](https://ollama.ai/), [oobabooga](https://github.com/oobabooga/text-generation-webui), [kobold.cpp](https://github.com/LostRuins/koboldcpp) |  [] |
|   - Cloud Deployment: AWS, GCP, Azure, SkyPilot, Specialized Hardware (TPUs) | [SkyPilot](https://github.com/skypilot-org/skypilot), [Hugging Face Inference API](https://huggingface.co/inference-api), [Together AI](https://www.together.ai/) |  [] |
|   - Serverless Functions, Edge Deployment (MLC LLM, mnn-llm)   | [AWS Lambda](https://aws.amazon.com/lambda/), [Google Cloud Functions](https://cloud.google.com/functions), [Azure Functions](https://azure.microsoft.com/en-us/services/functions/), [MLC LLM](https://github.com/mlc-ai/mlc-llm), [mnn-llm](https://github.com/wangzhaode/mnn-llm/blob/master/README_en.md) |  [] |
| - Inference Optimization:                               |           |           |
|   - Quantization: GPTQ, EXL2, GGUF, llama.cpp         | [Introduction to Quantization](https://mlabonne.github.io/blog/posts/Introduction_to_Weight_Quantization.html), [Quantization with GGUF and llama.cpp Notebook](https://colab.research.google.com/drive/1pL8k7m04mgE5jo2NrjGi8atB0j_37aDD?usp=sharing), [4-bit LLM Quantization with GPTQ](https://mlabonne.github.io/blog/posts/4_bit_Quantization_with_GPTQ.html), [4-bit Quantization using GPTQ Notebook](https://colab.research.google.com/drive/1lSvVDaRgqQp_mWK_jC9gydz6_-y6Aq4A?usp=sharing), [ExLlamaV2: The Fastest Library to Run LLMs](https://mlabonne.github.io/blog/posts/ExLlamaV2_The_Fastest_Library_to_Run%C2%A0LLMs.html), [ExLlamaV2 Notebook](https://colab.research.google.com/drive/1yrq4XBlxiA0fALtMoT2dwiACVc77PHou?usp=sharing), [AutoQuant Notebook](https://colab.research.google.com/drive/1b6nqC7UZVt8bx4MksX7s656GXPM-eWw4?usp=sharing) | [] |
|   - Flash Attention, Key-Value Cache (MQA, GQA)     | [Flash-Attention](https://github.com/Dao-AILab/flash-attention), [Multi-Query Attention](https://arxiv.org/abs/1911.02150), [Grouped-Query Attention](https://arxiv.org/abs/2305.13245) |  [] | 
|   - Knowledge Distillation, Pruning                   | [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531), [To prune, or not to prune: exploring the efficacy of pruning for model compression](https://arxiv.org/abs/1710.01878) |  [] |
|   - Speculative Decoding                              | [Hugging Face: Assisted Generation](https://huggingface.co/blog/assisted-generation) |  [] |
| - Building LLM Applications:                         |           |           |
|   - APIs: OpenAI, Google, Anthropic, Cohere, OpenRouter, Hugging Face | [OpenAI API](https://platform.openai.com/), [Google AI Platform](https://cloud.google.com/ai-platform/), [Anthropic API](https://docs.anthropic.com/claude/reference/getting-started-with-the-api), [Cohere API](https://docs.cohere.com/docs), [OpenRouter](https://openrouter.ai/), [Hugging Face Inference API](https://huggingface.co/inference-api) |  [] |
|   - Web Frameworks: Gradio, Streamlit                 | [Gradio](https://www.gradio.app/), [Streamlit](https://docs.streamlit.io/), [ZeroSpace Notebook](https://colab.research.google.com/drive/1LcVUW5wsJTO2NGmozjji5CkC--646LgC) |  [] |
|   - User Interfaces, Chatbots                        | [Chainlit](https://docs.chainlit.io/overview), [Langchain-Chatchat](https://github.com/chatchat-space/Langchain-Chatchat) |  [] |
| - MLOps for LLMs:                                      |           |           |
|   - CI/CD, Monitoring, Model Management               | [CometLLM](https://github.com/comet-ml/comet-llm), [MLflow](https://mlflow.org/), [Kubeflow](https://www.kubeflow.org/) |  [] |
|   - Experiment Tracking, Model Versioning             | [Weights & Biases](https://wandb.ai/site/solutions/llmops), [MLflow Tracking](https://www.mlflow.org/docs/latest/tracking.html) |  [] |
|   - Data & Model Pipelines                            | [ZenML](https://zenml.io/), [DVC](https://dvc.org/) |  [] |
| - LLM Security:                                         |           |           | 
|   - Prompt Hacking: Injection, Leaking, Jailbreaking  | [OWASP LLM Top 10](https://owasp.org/www-project-top-10-for-large-language-model-applications/), [Prompt Injection Primer](https://github.com/jthack/PIPE) |  [] |
|   - Backdoors: Data Poisoning, Trigger Backdoors     | [Trojaning Language Models for Fun and Profit](https://arxiv.org/abs/2008.00313), [Hidden Trigger Backdoor Attacks](https://arxiv.org/abs/1912.02257) |  [] |
|   - Defensive Measures: Red Teaming, Garak, Langfuse | [Red Teaming LLMs](https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/red-teaming), [garak](https://github.com/leondz/garak/), [Langfuse](https://github.com/langfuse/langfuse)  |  [] | 
