# 🚀 NLP Journey - Roadmap to Learn LLMs from Scratch with Modern NLP Methods in 2024 

Welcome! This repository is your comprehensive guide 🗺️ to mastering Natural Language Processing (NLP), from the fundamentals all the way to understanding and applying Large Language Models (LLMs). Whether you're a beginner or have some NLP experience, this roadmap will equip you with the practical skills needed for NLP and LLM-related roles in 2024 and beyond. We'll be using Jupyter Notebooks 📓 for hands-on practice along the way.

##  📚 Chapter 1: Foundations of NLP

Let's start by building a strong foundation in core NLP concepts:

**🌟 Core NLP Concepts**

| Topic                                       | Resources                                           | 
|---------------------------------------------|------------------------------------------------------|
| Introduction to NLP: Syntax, Semantics, Pragmatics, Discourse                      | [What is Natural Language Processing (NLP)?](https://www.datacamp.com/blog/what-is-natural-language-processing) | 

**⚙️ Text Preprocessing & Feature Engineering** 

| Topic                                                    | Resources                                                                                   | Practices |
|----------------------------------------------------------|----------------------------------------------------------------------------------------------|-----------|
| Tokenization (Word, Subword - BPE, SentencePiece)        |[Hugging Face Tokenizers](https://huggingface.co/docs/tokenizers/index) <br> [Tokenization, Lemmatization, Stemming, and Sentence Segmentation](https://colab.research.google.com/drive/18ZnEnXKLQkkJoBXMZR2rspkWSm9EiDuZ) <br> [Andrej Karpathy: Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE&t=1158s)|<a target="_blank" href="https://colab.research.google.com/github/mshojaei77/NLP-Journe"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>      |
| Stemming & Lemmatization                                | [Stanford: Stemming and lemmatization](https://nlp.stanford.edu/IR-book/html/htmledition/stemming-and-lemmatization-1.html) <br> [NLTK Stemming and Lemmatization](https://www.nltk.org/howto/stem.html)                         | []         |
| Stop Word Removal, Punctuation Handling                 | [NLTK Stop Words](https://www.nltk.org/book/ch02.html#stop-words-corpus)                         | []         |
| Bag-of-Words (BoW), TF-IDF, N-grams                      | [Scikit-learn: Text Feature Extraction](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) | []         |

**🌐 Word Embeddings**

| Topic                                                            | Resources                                                                                      | Practices |
|-----------------------------------------------------------------|-------------------------------------------------------------------------------------------------|-----------|
| Word2Vec, GloVe, FastText                                     | [Jay Alammar - Illustrated Word2Vec](https://jalammar.github.io/illustrated-word2vec/) <br> [Gensim Word2Vec](https://radimrehurek.com/gensim/models/word2vec.html) <br> [Stanford GloVe](https://nlp.stanford.edu/projects/glove/) | []         |
| Contextual Embeddings (ELMo, BERT)                            |  [Stanford NLP: N-gram Language Models](https://nlp.stanford.edu/fsnlp/lm.html)                  | []         |


## 🤖 Chapter 2: Essential NLP Tasks & Algorithms

Now, let's dive into common NLP tasks and the algorithms that power them:

| Topic                                                    | Resources                                                                                          | Practices |
|----------------------------------------------------------|----------------------------------------------------------------------------------------------------|-----------|
| Text Classification (Naive Bayes, SVM, Logistic Regression, Deep Learning) | [Scikit-learn Text Classification](https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html) <br> [Hugging Face Text Classification](https://huggingface.co/docs/transformers/tasks/sequence_classification) <br> [FastText](https://github.com/facebookresearch/fastText) |  [] |
| Sentiment Analysis (Lexicon-Based, Machine Learning, Aspect-Based) | [NLTK Sentiment Analysis](https://www.nltk.org/howto/sentiment.html) <br> [TextBlob Sentiment Analysis](https://textblob.readthedocs.io/en/dev/quickstart.html#sentiment-analysis) <br> [VADER Sentiment Analysis](https://github.com/cjhutto/vaderSentiment) |  [] |
| Named Entity Recognition (NER) (NLTK, spaCy, Transformers) | [NLTK NER](https://www.nltk.org/book/ch07.html) <br> [spaCy NER](https://spacy.io/usage/linguistic-features#named-entities) <br> [Hugging Face NER](https://huggingface.co/docs/transformers/tasks/token-classification) <br> [MIT Information Extraction Toolkit](https://github.com/mit-nlp/MITIE) |  [] |
| Text Clustering (K-Means, Hierarchical Clustering, DBSCAN, OPTICS) | [Scikit-learn Clustering](https://scikit-learn.org/stable/modules/clustering.html)                   |  [] |
| Topic Modeling (LDA, NMF)                                 | [Gensim Topic Modeling](https://radimrehurek.com/gensim/auto_examples/tutorials/run_lda.html) <br> [Scikit-learn NMF](https://scikit-learn.org/stable/modules/decomposition.html#non-negative-matrix-factorization-nmf-or-nnmf) <br> [BigARTM](https://github.com/bigartm/bigartm) |  [] |
| Information Retrieval (TF-IDF, BM25, Query Expansion, Vector Search, Semantic Search) | [Elasticsearch](https://www.elastic.co/) <br> [Solr](https://lucene.apache.org/solr/) <br> [Pinecone](https://www.pinecone.io/) |  [] |
| Question Answering                                          | [DrQA](https://github.com/facebookresearch/DrQA) <br> [Document-QA](https://github.com/allenai/document-qa)                              | []         |
| Knowledge Extraction                                      | [Template-Based Information Extraction without the Templates](https://www.usna.edu/Users/cs/nchamber/pubs/acl2011-chambers-templates.pdf) <br> [Privee: An Architecture for Automatically Analyzing Web Privacy Policies](https://www.sebastianzimmeck.de/zimmeckAndBellovin2014Privee.pdf) <br> [LEGALO](https://link.springer.com/chapter/10.1007/978-3-319-49001-4_16)  | []       |


**🚀 NLP Applications** 

Let's explore how these tasks translate into real-world NLP applications:

| Topic            | Resources                                                              | Practices |
|-----------------|--------------------------------------------------------------------------|-----------|
| Dialogue Systems | [Chat script](https://github.com/bwilcox-1234/ChatScript) <br> [Chatter bot](http://chatterbot.readthedocs.io/en/stable/#) <br> [RiveScript](https://www.rivescript.com/about) <br> [SuperScript](http://superscriptjs.com/) <br> [BotKit](https://github.com/howdyai/botkit) | []         | 
| Machine Translation | [Berkeley Aligner](https://code.google.com/p/berkeleyaligner/) <br> [cdec](https://github.com/redpony/cdec) <br> [Jane](http://www-i6.informatik.rwth-aachen.de/jane/) <br> [Joshua](http://joshua-decoder.org/) <br> [Moses](http://www.statmt.org/moses/) <br> [alignment-with-openfst](https://github.com/ldmt-muri/alignment-with-openfst) <br> [zmert](http://cs.jhu.edu/~ozaidan/zmert/)  | [] |
| Text Summarization | [IndoSum](https://github.com/kata-ai/indosum) <br> [Cohere Summarize Beta](https://txt.cohere.ai/summarize-beta/) | [] |


##  🧠 Chapter 3: Deep Learning for NLP

Time to unleash the power of deep learning for NLP tasks:

**🧠 Neural Network Fundamentals**

| Topic                              | Resources                                                                                                                                            | Practices |
|------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------|-----------|
| Neural Network Basics, Backpropagation | [3Blue1Brown - Neural Networks](https://www.youtube.com/watch?v=aircAruvnKk) <br> [freeCodeCamp - Deep Learning Crash Course](https://www.youtube.com/watch?v=VyWAvY2CF9c) |  [] |
| Perceptron                        | [3Blue1Brown - Neural Networks](https://www.youtube.com/watch?v=aircAruvnKk)                                                                         |  [] |

**🛠️ Deep Learning Frameworks**

| Topic                         | Resources                                                                                    | Practices |
|---------------------------------|---------------------------------------------------------------------------------------------|-----------|
| PyTorch, JAX, TensorFlow       | [PyTorch Tutorials](https://pytorch.org/tutorials/) <br> [JAX Documentation](https://jax.readthedocs.io/en/latest/) <br> [TensorFlow Tutorials](https://www.tensorflow.org/tutorials) <br> [Caffe](http://arxiv.org/pdf/1409.3215v1.pdf) |  [] |
| MxNet, Numpy                  | [MxNet + Numpy]( https://github.com/dmlc/minpy)                                                 | []         |

**⚙️ Deep Learning Architectures for NLP**

| Topic                                                       | Resources                                                                                                                                                                          | Practices |
|------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------|
| Recurrent Neural Networks (RNNs) (Sequence Modeling, LSTMs, GRUs, Attention) | [colah's blog: Understanding LSTMs](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) <br> [Andrej Karpathy: The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) <br> [Bayesian Recurrent Neural Network for Language Modeling](http://chien.cm.nctu.edu.tw/bayesian-recurrent-neural-network-for-language-modeling/) <br> [RNNLM](http://www.fit.vutbr.cz/~imikolov/rnnlm/) <br>  [KALDI LSTM](https://github.com/dophist/kaldi-lstm) |  [] |
| Convolutional Neural Networks (CNNs) for Text  (Classification, Hierarchical CNNs) | [Understanding Convolutional Neural Networks for NLP](http://www.wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp/) <br> [Kim Yoon: Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882) |  [] |
| Sequence-to-Sequence Models (Attention, Transformers, T5, BART) | [Jay Alammar: The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) <br> [Google AI Blog: Transformer Networks](https://ai.googleblog.com/2017/08/transformer-networks-state-of-art.html) <br> [Hugging Face: T5](https://huggingface.co/docs/transformers/model_doc/t5) <br> [Hugging Face: BART](https://huggingface.co/docs/transformers/model_doc/bart) |  [] |


##  🚀 Chapter 4: Large Language Models (LLMs)

This chapter delves into the exciting world of LLMs:

**🤖 The Transformer Architecture**

| Topic                                                            | Resources                                                                                                         | Practices |
|-----------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------|-----------|
| Attention, Residual Connections, Layer Normalization, RoPE        | [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) <br> [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/) <br> [Visual Intro to Transformers](https://www.youtube.com/watch?v=wjZofJX0v4M&t=187s) <br> [LLM Visualization](https://bbycroft.net/llm) <br> [nanoGPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) <br> [GPT in 60 Lines of NumPy](https://jaykmody.com/blog/gpt-from-scratch/) |  [] |

**🏗️ LLM Architectures, Pre-training, & Post-training**

| Topic                                                                       | Resources                                                                                                                                                                                                                    | Practices |
|---------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------|
| GPT, BERT, T5, Llama, PaLM, Phi-3                                         | [LLMDataHub](https://github.com/Zjh-819/LLMDataHub) <br> [Hugging Face: Causal Language Modeling](https://huggingface.co/learn/nlp-course/chapter7/6?fw=pt) <br> [TinyLlama](https://github.com/jzhang38/TinyLlama) <br> [Chinchilla's Wild Implications](www.lesswrong.com/posts/6Fpvch8RR29qLEWNH/chinchilla-s-wild-implications) <br> [BLOOM](https://bigscience.notion.site/BLOOM-BigScience-176B-Model-ad073ca07cdf479398d5f95d88e218c4) <br> [OPT-175 Logbook](https://github.com/facebookresearch/metaseq/blob/main/projects/OPT/chronicles/OPT175B_Logbook.pdf) <br> [LLM 360](https://www.llm360.ai/) <br> [New LLM Pre-training and Post-training Paradigms](https://magazine.sebastianraschka.com/p/new-llm-pre-training-and-post-training) <br> [Phi-3CookBook](https://github.com/microsoft/Phi-3CookBook) <br> [Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling](https://arxiv.org/abs/2304.01373) <br> [LLM Reading List](https://github.com/crazyofapple/Reading_groups/) <br>  [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/pdf/2312.00752) <br> [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434) <br>  [Jamba: A Hybrid Transformer-Mamba Language Model](https://arxiv.org/pdf/2403.19887) <br> [Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality](https://arxiv.org/abs/2405.21060) <br> [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783) |  [] |
| Emerging Architectures (DeepSeek-v2, Jamba, Mixture of Experts - MoE) | [DeepSeek-v2](https://arxiv.org/abs/2405.04434) <br> [Jamba](https://arxiv.org/abs/2403.19887) <br> [Hugging Face: Mixture of Experts Explained](https://huggingface.co/blog/moe) <br> [Create MoEs with MergeKit Notebook](https://colab.research.google.com/drive/1obulZ1ROXHjYLn6PPZJwRR6GzgQogxxb?usp=sharing)  <br> [GLaM: Efficient Scaling of Language Models with Mixture-of-Experts](https://arxiv.org/pdf/2112.06905.pdf) <br> [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/pdf/2101.03961.pdf) |  [] |

**🔧 Fine-tuning & Adapting LLMs** 

Learn to customize LLMs for your specific needs:

| Topic                                                                    | Resources                                                                                                                                                                                               | Practices |
|-------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------|
| Supervised Fine-Tuning (SFT)                                             | [Fine-Tune Your Own Llama 2 Model](https://mlabonne.github.io/blog/posts/Fine_Tune_Your_Own_Llama_2_Model_in_a_Colab_Notebook.html) <br> [Padding Large Language Models](https://towardsdatascience.com/padding-large-language-models-examples-with-llama-2-199fb10df8ff) <br> [A Beginner's Guide to LLM Fine-Tuning](https://mlabonne.github.io/blog/posts/A_Beginners_Guide_to_LLM_Finetuning.html) <br> [unslothai](https://github.com/unslothai/unsloth) <br> [Fine-tune Llama 2 with QLoRA Notebook](https://colab.research.google.com/drive/1PEQyJO1-f6j0S_XJ8DV50NkpzasXkrzd?usp=sharing) <br> [Fine-tune CodeLlama using Axolotl Notebook](https://colab.research.google.com/drive/1Xu0BrCB7IShwSWKVcfAfhehwjDrDMH5m?usp=sharing) <br> [Fine-tune Mistral-7b with QLoRA Notebook](https://colab.research.google.com/drive/1o_w0KastmEJNVwT5GoqMCciH-18ca5WS?usp=sharing) <br> [Fine-tune Mistral-7b with DPO Notebook](https://colab.research.google.com/drive/15iFBr1xWgztXvhrj5I9fBv20c7CFOPBE?usp=sharing) <br> [Fine-tune Llama 3 with ORPO Notebook](https://colab.research.google.com/drive/1eHNWg9gnaXErdAa8_mcvjMupbSS6rDvi) <br> [Fine-tune Llama 3.1 with Unsloth Notebook](https://colab.research.google.com/drive/164cg_O7SV7G8kZr_JXqLd6VC7pd86-1Z?usp=sharing) <br> [flux-finetune](https://github.com/gradient-ai/flux-finetune) <br> [torchtune](https://github.com/pytorch/torchtune) <br> [Flan Collection: Designing Data and Methods for Effective Instruction Tuning](https://arxiv.org/pdf/2301.13688.pdf) | []         |
| Parameter-Efficient Fine-tuning (PEFT) (LoRA, Adapters, Prompt Tuning)  | [LoRA Insights](https://lightning.ai/pages/community/lora-insights/) <br> [Hugging Face: Parameter-Efficient Fine-Tuning](https://huggingface.co/blog/peft) <br> [FLAN](https://openreview.net/forum?id=gEZrGCozdqR) <br> [T0](https://arxiv.org/abs/2110.08207)                                                 |  [] |
| Reinforcement Learning from Human Feedback (RLHF) (PPO, DPO)              | [Distilabel](https://github.com/argilla-io/distilabel) <br> [An Introduction to Training LLMs using RLHF](https://wandb.ai/ayush-thakur/Intro-RLAIF/reports/An-Introduction-to-Training-LLMs-Using-Reinforcement-Learning-From-Human-Feedback-RLHF---VmlldzozMzYyNjcy) <br> [Hugging Face: Illustration RLHF](https://huggingface.co/blog/rlhf) <br> [Hugging Face: Preference Tuning LLMs](https://huggingface.co/blog/pref-tuning) <br> [LLM Training: RLHF and Its Alternatives](https://magazine.sebastianraschka.com/p/llm-training-rlhf-and-its-alternatives) <br> [Fine-tune Mistral-7b with DPO](https://mlabonne.github.io/blog/posts/Fine_tune_Mistral_7b_with_DPO.html) <br> [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/pdf/2305.18290.pdf) <br> [Training language models to follow instructions with human feedback](https://arxiv.org/pdf/2203.02155.pdf) <br> [WebGPT: Browser-assisted question-answering with human feedback](https://www.semanticscholar.org/paper/WebGPT%3A-Browser-assisted-question-answering-with-Nakano-Hilton/2f3efe44083af91cef562c1a3451eee2f8601d22) <br> [Improving alignment of dialogue agents via targeted human judgements](https://arxiv.org/pdf/2209.14375.pdf) <br> [OPT-IML: Scaling Language Model Instruction Meta Learning through the Lens of Generalization](https://arxiv.org/pdf/2212.12017) |  [] |
| Model Merging (SLERP, DARE/TIES, FrankenMoEs)                              | [Merge LLMs with mergekit](https://mlabonne.github.io/blog/posts/2024-01-08_Merge_LLMs_with_mergekit.html) <br> [DARE/TIES](https://arxiv.org/abs/2311.03099) <br> [Phixtral](https://huggingface.com/mlabonne/phixtral-2x2_8) <br> [MergeKit](https://github.com/cg123/mergekit) <br> [Merge LLMs with MergeKit Notebook](https://colab.research.google.com/drive/1_JS7JKJAQozD48-LhYdegcuuZ2ddgXfr?usp=sharing) <br> [LazyMergekit Notebook](https://colab.research.google.com/drive/1obulZ1ROXHjYLn6PPZJwRR6GzgQogxxb?usp=sharing)  |  [] |

**📊 LLM Evaluation** 

How do we know if an LLM is performing well?

| Topic                      | Resources                                                                                                                                            | Practices |
|----------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------|-----------|
| LLM Evaluation Benchmarks & Tools | [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) <br> [MixEval](https://github.com/Psycoy/MixEval) <br> [lighteval](https://github.com/huggingface/lighteval) <br> [OLMO-eval](https://github.com/allenai/OLMo-Eval) <br> [instruct-eval](https://github.com/declare-lab/instruct-eval) <br> [simple-evals](https://github.com/openai/simple-evals) <br> [Giskard](https://github.com/Giskard-AI/giskard) <br> [LangSmith](https://www.langchain.com/langsmith)  <br>  [Ragas](https://github.com/explodinggradients/ragas) <br> [Chatbot Arena Leaderboard](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard) <br> [MixEval Leaderboard](https://mixeval.github.io/#leaderboard) <br> [AlpacaEval Leaderboard](https://tatsu-lab.github.io/alpaca_eval/) <br> [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) <br> [OpenCompass 2.0 LLM Leaderboard](https://rank.opencompass.org.cn/leaderboard-llm-v2) <br> [Berkeley Function-Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html) <br> [HELM](https://arxiv.org/pdf/2211.09110.pdf) <br> [BIG-bench](https://github.com/google/BIG-bench) | []         |

**🗣️ Prompt Engineering**

The art of crafting effective prompts for LLMs:

| Topic                                                    | Resources                                                                                                                                       | Practices |
|----------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------|-----------|
| Prompt Engineering Techniques (Zero-Shot, Few-Shot, Chain-of-Thought, ReAct)  | [Prompt Engineering Guide](https://www.promptingguide.ai/) <br> [Lilian Weng: Prompt Engineering](https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/) <br> [LLM Prompt Engineering Simplified Book](https://llmnanban.akmmusai.pro/Book/LLM-Prompt-Engineering-Simplified-Book/) <br> [Chain-of-Thoughts Papers](https://github.com/Timothyxxx/Chain-of-ThoughtsPapers) <br> [Awesome Deliberative Prompting](https://github.com/logikon-ai/awesome-deliberative-prompting) <br> [Instruction-Tuning-Papers](https://github.com/SinclairCoder/Instruction-Tuning-Papers)  <br>  [Tree of Thoughts: Deliberate Problem Solving with Large Language Models](https://arxiv.org/pdf/2305.10601.pdf) <br> [Awesome ChatGPT Prompts](https://github.com/f/awesome-chatgpt-prompts) <br> [awesome-chatgpt-prompts-zh](https://github.com/PlexPt/awesome-chatgpt-prompts-zh)   |  [] |
| Task-Specific Prompting (e.g., Code Generation)         | [Code Generation with AlphaCodium: From Prompt Engineering to Flow Engineering](https://arxiv.org/abs/2401.08500) <br> [Codex](https://arxiv.org/pdf/2107.03374.pdf)                                |  [] |
| Structuring LLM Outputs (Templates, JSON, LMQL, Outlines, Guidance) | [Chat Template](https://huggingface.co/blog/chat-templates) <br> [Outlines - Quickstart](https://outlines-dev.github.io/outlines/quickstart/) <br> [LMQL - Overview](https://lmql.ai/docs/language/overview.html) <br> [Microsoft Guidance](https://github.com/microsoft/guidance) <br> [Guidance](https://github.com/microsoft/guidance) <br> [Outlines](https://github.com/normal-computing/outlines) |  [] |

**🗃️ Retrieval Augmented Generation (RAG)**

| Topic                                                    | Resources                                                                                                                      | Practices |
|----------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------|-----------|
|   Document Loaders, Text Splitters                  | [LangChain Text Splitters](https://python.langchain.com/docs/modules/data_connection/document_transformers/) <br> [LlamaIndex Data Connectors](https://gpt-index.readthedocs.io/en/latest/guides/primer/data_connectors.html) |  [] |
|   Embedding Models                                    | [Sentence Transformers Library](https://www.sbert.net/) <br> [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) <br> [InferSent](https://github.com/facebookresearch/InferSent)          |  [] |
|   Vector Databases (Chroma, Pinecone, Milvus, FAISS, Annoy) | [Chroma](https://www.trychroma.com/) <br> [Pinecone](https://www.pinecone.io/) <br> [Milvus](https://milvus.io/) <br> [FAISS](https://faiss.ai/) <br> [Annoy](https://github.com/spotify/annoy)  |  [] |
|   Orchestrators (LangChain, LlamaIndex, FastRAG)       | [LangChain](https://python.langchain.com/) <br> [LlamaIndex](https://docs.llamaindex.ai/en/stable/) <br> [FastRAG](https://github.com/IntelLabs/fastRAG) <br> [🦜🔗 Awesome LangChain](https://github.com/kyrolabs/awesome-langchain) |  [] |
|   Query Expansion, Re-ranking, HyDE                    | [HyDE](https://arxiv.org/abs/2212.10496) <br> [LangChain Retrievers](https://python.langchain.com/docs/modules/data_connection/retrievers/)                              |  [] |
|   RAG Fusion                                           | [RAG-fusion](https://github.com/Raudaschl/rag-fusion)                                                                         |  [] |
|   Evaluation (Context Precision/Recall, Faithfulness, Relevancy, Ragas, DeepEval) | [Ragas](https://github.com/explodinggradients/ragas/tree/main) <br> [DeepEval](https://github.com/confident-ai/deepeval)                |  [] |
|   Query Construction (SQL, Cypher)                     | [LangChain Query Construction](https://blog.langchain.dev/query-construction/) <br> [LangChain SQL](https://python.langchain.com/docs/use_cases/qa_structured/sql) |  [] |
|   Agents & Tools (Google Search, Wikipedia, Python, Jira) | [LangChain Agents](https://python.langchain.com/docs/modules/agents/)                                                                |  [] |
|   Programmatic LLMs (DSPy)                           | [DSPy](https://github.com/stanfordnlp/dspy) <br> [dspy](https://github.com/stanfordnlp/dspy)                                                                                    |  [] |


## 🎨 Chapter 5: Multimodal Learning & Applications

Go beyond text and explore the world of multimodal LLMs:

| Topic                                                    | Resources                                                                                                                    | Practices |
|----------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------|-----------|
| Multimodal LLMs (CLIP, ViT, LLaVA, MiniCPM-V, GPT-SoVITS)                       | [OpenAI CLIP](https://openai.com/research/clip) <br> [Google AI Blog: ViT](https://ai.googleblog.com/2020/10/an-image-is-worth-16x16-words.html) <br> [LLaVA](https://llava-vl.github.io/) <br> [MiniCPM-V 2.6](https://github.com/OpenBMB/MiniCPM-V) <br> [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)  |  [] |
| Vision-Language Tasks (Image Captioning, VQA, Visual Reasoning) | [Hugging Face: Vision-Language Tasks](https://huggingface.co/docs/transformers/tasks/vision-language-modeling) <br> [Microsoft Kosmos-1](https://arxiv.org/abs/2302.14045) <br> [Google PaLM-E](https://palm-e.github.io/) <br> [Visual Instruction Tuning](https://arxiv.org/abs/2304.08485)                      |  [] |
| Text-to-Image Generation, Video Understanding           | [Stability AI: Stable Diffusion](https://stability.ai/stable-image) <br> [OpenAI DALL-E 2](https://openai.com/dall-e-2) <br> [Hugging Face: Video Understanding](https://huggingface.co/docs/transformers/tasks/video-classification) <br> [Deep-Live-Cam](https://github.com/hacksider/Deep-Live-Cam)  |  [] |
| Emerging Trends (Neuro-Symbolic AI, LLMs for Robotics)  |                                                                                                                                 | []         |

## 💻 Chapter 6: Deployment & Productionizing LLMs

Learn how to take your LLM models from experimentation to the real world:

**🚀 Deployment Strategies**

| Topic                                                             | Resources                                                                                                                                            | Practices |
|------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------|-----------|
| Local Servers (LM Studio, Ollama, Oobabooga, Kobold.cpp)          | [LM Studio](https://lmstudio.ai/) <br> [Ollama](https://ollama.ai/) <br> [oobabooga](https://github.com/oobabooga/text-generation-webui) <br> [kobold.cpp](https://github.com/LostRuins/koboldcpp) <br> [llama.cpp](https://github.com/ggerganov/llama.cpp) <br> [mistral.rs](https://github.com/EricLBuehler/mistral.rs) <br> [Serge](https://github.com/serge-chat/serge) |  [] |
| Cloud Deployment (AWS, GCP, Azure, SkyPilot, Specialized Hardware (TPUs)) | [SkyPilot](https://github.com/skypilot-org/skypilot) <br> [Hugging Face Inference API](https://huggingface.co/inference-api) <br> [Together AI](https://www.together.ai/) <br> [Modal](https://modal.com/docs/guide/ex/potus_speech_qanda) <br> [Metal](https://getmetal.io/)           |  [] |
| Serverless Functions, Edge Deployment (MLC LLM, mnn-llm)          | [AWS Lambda](https://aws.amazon.com/lambda/) <br> [Google Cloud Functions](https://cloud.google.com/functions) <br> [Azure Functions](https://azure.microsoft.com/en-us/services/functions/) <br> [MLC LLM](https://github.com/mlc-ai/mlc-llm) <br> [mnn-llm](https://github.com/wangzhaode/mnn-llm/blob/master/README_en.md)  |  [] |
| LLM Serving                                                        | [LitServe](https://github.com/Lightning-AI/LitServe) <br> [vLLM](https://github.com/vllm-project/vllm) <br> [TGI](https://huggingface.co/docs/text-generation-inference/en/index) <br> [FastChat](https://github.com/lm-sys/FastChat) <br> [Jina](https://github.com/jina-ai/langchain-serve) <br> [LangServe](https://github.com/langchain-ai/langserve)                                             |           |

** Inference Optimization**

| Topic                                              | Resources                                                                                                                                                                                                                  | Practices |
|---------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------|
| Quantization (GPTQ, EXL2, GGUF, llama.cpp, exllama)          | [Introduction to Quantization](https://mlabonne.github.io/blog/posts/Introduction_to_Weight_Quantization.html) <br> [Quantization with GGUF and llama.cpp Notebook](https://colab.research.google.com/drive/1pL8k7m04mgE5jo2NrjGi8atB0j_37aDD?usp=sharing) <br> [4-bit LLM Quantization with GPTQ](https://mlabonne.github.io/blog/posts/4_bit_Quantization_with_GPTQ.html) <br> [4-bit Quantization using GPTQ Notebook](https://colab.research.google.com/drive/1lSvVDaRgqQp_mWK_jC9gydz6_-y6Aq4A?usp=sharing) <br> [ExLlamaV2: The Fastest Library to Run LLMs](https://mlabonne.github.io/blog/posts/ExLlamaV2_The_Fastest_Library_to_Run%C2%A0LLMs.html) <br> [ExLlamaV2 Notebook](https://colab.research.google.com/drive/1yrq4XBlxiA0fALtMoT2dwiACVc77PHou?usp=sharing) <br> [AutoQuant Notebook](https://colab.research.google.com/drive/1b6nqC7UZVt8bx4MksX7s656GXPM-eWw4?usp=sharing) <br> [exllama](https://github.com/turboderp/exllama) | []         |
| Flash Attention, Key-Value Cache (MQA, GQA)        | [Flash-Attention](https://github.com/Dao-AILab/flash-attention) <br> [Multi-Query Attention](https://arxiv.org/abs/1911.02150) <br> [Grouped-Query Attention](https://arxiv.org/abs/2305.13245)      |  [] |
| Knowledge Distillation, Pruning                    | [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531) <br> [To prune, or not to prune: exploring the efficacy of pruning for model compression](https://arxiv.org/abs/1710.01878) |  [] |
| Speculative Decoding                               | [Hugging Face: Assisted Generation](https://huggingface.co/blog/assisted-generation)                                                                                                     |  [] |

** Building with LLMs**

| Topic                                                     | Resources                                                                                                                                             | Practices |
|----------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------|-----------|
| APIs (OpenAI, Google, Anthropic, Cohere, OpenRouter, Hugging Face) | [OpenAI API](https://platform.openai.com/) <br> [Google AI Platform](https://cloud.google.com/ai-platform/) <br> [Anthropic API](https://docs.anthropic.com/claude/reference/getting-started-with-the-api) <br> [Cohere API](https://docs.cohere.com/docs) <br> [OpenRouter](https://openrouter.ai/) <br> [Hugging Face Inference API](https://huggingface.co/inference-api) <br> [GPTRouter](https://gpt-router.writesonic.com/) |  [] |
| Web Frameworks (Gradio, Streamlit)                        | [Gradio](https://www.gradio.app/) <br> [Streamlit](https://docs.streamlit.io/) <br> [ZeroSpace Notebook](https://colab.research.google.com/drive/1LcVUW5wsJTO2NGmozjji5CkC--646LgC)                          |  [] |
| User Interfaces, Chatbots                               | [Chainlit](https://docs.chainlit.io/overview) <br> [Langchain-Chatchat](https://github.com/chatchat-space/Langchain-Chatchat) <br> [llm-ui](https://github.com/llm-ui-kit/llm-ui)                               |  [] |
| End-to-End LLM Projects |  [Awesome NLP Projects](https://github.com/EugeniuCostezki/awesome-nlp-projects)  | [] |
| LLM Application Frameworks | [LangChain](https://github.com/hwchase17/langchain) <br> [Haystack](https://haystack.deepset.ai/) <br> [Semantic Kernel](https://github.com/microsoft/semantic-kernel) <br> [LlamaIndex](https://github.com/jerryjliu/llama_index) <br> [LMQL](https://lmql.ai) <br> [ModelFusion](https://github.com/lgrammel/modelfusion) <br> [Flappy](https://github.com/pleisto/flappy) <br> [LiteChain](https://github.com/rogeriochaves/litechain) <br> [magentic](https://github.com/jackmpcollins/magentic) | [] |

** MLOps for LLMs**

| Topic                                          | Resources                                                                                                                          | Practices |
|------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------|-----------|
| CI/CD, Monitoring, Model Management            | [CometLLM](https://github.com/comet-ml/comet-llm) <br> [MLflow](https://mlflow.org/) <br> [Kubeflow](https://www.kubeflow.org/) <br> [Evidently](https://github.com/evidentlyai/evidently) <br> [Arthur Shield](https://www.arthur.ai/get-started)  <br>  [Mona](https://github.com/monalabs/mona-openai) <br> [Openllmetry](https://github.com/traceloop/openllmetry) <br> [Graphsignal](https://graphsignal.com/) <br> [Arize-Phoenix](https://phoenix.arize.com/)                |  [] |
| Experiment Tracking, Model Versioning          | [Weights & Biases](https://wandb.ai/site/solutions/llmops) <br> [MLflow Tracking](https://www.mlflow.org/docs/latest/tracking.html) |  [] |
| Data & Model Pipelines                       | [ZenML](https://zenml.io/) <br> [DVC](https://dvc.org/)                                                                                   |  [] |

** LLM Security**

| Topic                                             | Resources                                                                                                                                  | Practices |
|---------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------|-----------|
| Prompt Hacking (Injection, Leaking, Jailbreaking) | [OWASP LLM Top 10](https://owasp.org/www-project-top-10-for-large-language-model-applications/) <br> [Prompt Injection Primer](https://github.com/jthack/PIPE) <br> [Awesome LLM Security](https://github.com/corca-ai/awesome-llm-security) |  [] |
| Backdoors (Data Poisoning, Trigger Backdoors)      | [Trojaning Language Models for Fun and Profit](https://arxiv.org/abs/2008.00313) <br> [Hidden Trigger Backdoor Attacks](https://arxiv.org/abs/1912.02257) |  [] |
| Defensive Measures (Red Teaming, Garak, Langfuse)  | [Red Teaming LLMs](https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/red-teaming) <br> [garak](https://github.com/leondz/garak/) <br> [Langfuse](https://github.com/langfuse/langfuse) |  [] | 

Let's embark on this NLP journey together! 
