# Gen AI Engineer - 3-Week Intensive Learning Plan

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-FFD21E?logo=huggingface&logoColor=black)
![LangChain](https://img.shields.io/badge/LangChain-Framework-1C3C3C?logo=langchain&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

**A structured, hands-on path from Python developer to Gen AI Engineer in 21 days.**

[Week 1: Foundations](#week-1-foundations---deep-learning-transformers--llms) | [Week 2: Core Skills](#week-2-core-genai-skills---rag-agents--chains) | [Week 3: Production](#week-3-production-skills-advanced-projects--portfolio) | [Resources](#-quick-reference-all-free-resources) | [Projects](#-project-summary)

</div>

---

## Prerequisites

| Requirement | Details |
|:------------|:--------|
| **Languages** | Python (comfortable with classes, functions, data structures) |
| **Libraries** | Pandas, Matplotlib, NumPy (basic usage) |
| **Time** | Full day (~10-12 hours/day) for 3 weeks |
| **Approach** | Learn concepts in the morning, build projects in the afternoon |

---

## Progress Tracker

| Week | Day | Topic | Status |
|:----:|:---:|:------|:------:|
| 1 | [Day 1](week1/day01_neural_networks_backpropagation/) | Neural Networks & Backpropagation | ✅ |
| 1 | [Day 2](week1/day02_language_modeling_nlp/) | Language Modeling & NLP Basics | ✅ |
| 1 | [Day 3](week1/day03_transformers/) | Transformers - The Core of All GenAI | ✅ |
| 1 | [Day 4](week1/day04_tokenization_huggingface/) | Tokenization & Hugging Face Basics | ✅ |
| 1 | [Day 5](week1/day05_prompt_engineering_llm_apis/) | Prompt Engineering & LLM APIs | ⬜ |
| 1 | [Day 6](week1/day06_embeddings_vector_databases/) | Embeddings & Vector Databases | ⬜ |
| 1 | [Day 7](week1/day07_review_consolidation/) | Review & Consolidation | ⬜ |
| 2 | [Day 8](week2/day08_langchain_fundamentals/) | LangChain Fundamentals | ⬜ |
| 2 | [Day 9](week2/day09_rag_basics/) | RAG - Basic Implementation | ⬜ |
| 2 | [Day 10](week2/day10_rag_advanced_youtube_project/) | RAG - Advanced + YouTube Project | ⬜ |
| 2 | [Day 11](week2/day11_ai_agents_react_pattern/) | AI Agents - Tool Use & ReAct | ⬜ |
| 2 | [Day 12](week2/day12_langgraph_multi_step_agents/) | LangGraph & Multi-Step Agents | ⬜ |
| 2 | [Day 13](week2/day13_fine_tuning_llms/) | Fine-Tuning LLMs | ⬜ |
| 2 | [Day 14](week2/day14_review_refine_integrate/) | Review, Refine & Integrate | ⬜ |
| 3 | [Day 15](week3/day15_multi_agent_systems/) | Multi-Agent Systems | ⬜ |
| 3 | [Day 16](week3/day16_agentic_rag/) | Agentic RAG & Advanced Patterns | ⬜ |
| 3 | [Day 17](week3/day17_deployment_apis/) | Deployment & APIs | ⬜ |
| 3 | [Day 18](week3/day18_evaluation_observability/) | Evaluation & Observability | ⬜ |
| 3 | [Day 19](week3/day19_multimodal_ai_vision/) | Multimodal AI & Vision | ⬜ |
| 3 | [Day 20](week3/day20_ethics_safety/) | Ethics, Safety & Responsible AI | ⬜ |
| 3 | [Day 21](week3/day21_portfolio_polish/) | Portfolio Day - Final Polish | ⬜ |

---

## Week 1: Foundations - Deep Learning, Transformers & LLMs

> Understand how neural networks and LLMs actually work under the hood. Without this, everything else is just API calls with no understanding.

---

### [Day 1: Neural Networks & Backpropagation](week1/day01_neural_networks_backpropagation/)

<table>
<tr><td><strong>Morning (4 hrs) - Learn</strong></td></tr>
<tr><td>

- Andrej Karpathy - [The spelled-out intro to neural networks and backpropagation: building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0)
- Follow along and code micrograd yourself

</td></tr>
<tr><td><strong>Afternoon (4 hrs) - Build</strong></td></tr>
<tr><td>

- Rebuild micrograd from scratch without looking at the video
- Extend it: add `tanh`, `relu`, `sigmoid` activation functions
- Train a small classifier on a toy dataset

</td></tr>
<tr><td><strong>Resources</strong></td></tr>
<tr><td>

- [micrograd GitHub repo](https://github.com/karpathy/micrograd)

</td></tr>
</table>

**Code:** [Book](week1/day01_neural_networks_backpropagation/README.md) | [micrograd.py](week1/day01_neural_networks_backpropagation/micrograd.py) | [exercises.py](week1/day01_neural_networks_backpropagation/exercises.py) | [visualizations.py](week1/day01_neural_networks_backpropagation/visualizations.py)

---

### [Day 2: Language Modeling & NLP Basics](week1/day02_language_modeling_nlp/)

<table>
<tr><td><strong>Morning (5 hrs) - Learn</strong></td></tr>
<tr><td>

- Karpathy - [The spelled-out intro to language modeling: building makemore](https://www.youtube.com/watch?v=PaCmpygFfXo)
- Karpathy - [Building makemore Part 2: MLP](https://www.youtube.com/watch?v=TCH_1BHY58I)

</td></tr>
<tr><td><strong>Afternoon (4 hrs) - Build</strong></td></tr>
<tr><td>

- Build a character-level name generator from scratch
- Train it on a dataset of names
- Experiment: change hidden layer sizes, learning rates, plot loss curves with matplotlib

</td></tr>
<tr><td><strong>Resources</strong></td></tr>
<tr><td>

- [makemore GitHub repo](https://github.com/karpathy/makemore)

</td></tr>
</table>

**Code:** [Book](week1/day02_language_modeling_nlp/README.md) | [bigram.py](week1/day02_language_modeling_nlp/bigram.py) | [bigram_neural.py](week1/day02_language_modeling_nlp/bigram_neural.py) | [mlp_lm.py](week1/day02_language_modeling_nlp/mlp_lm.py)

---

### [Day 3: Transformers - The Core of All GenAI](week1/day03_transformers/)

<table>
<tr><td><strong>Morning (5 hrs) - Learn</strong></td></tr>
<tr><td>

- Karpathy - [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- Jay Alammar - [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)

</td></tr>
<tr><td><strong>Afternoon (5 hrs) - Build</strong></td></tr>
<tr><td>

- Code a mini GPT from scratch following Karpathy's video
- Train it on a text file of your choice (Shakespeare, song lyrics, etc.)
- Generate text and observe how quality changes with training

</td></tr>
<tr><td><strong>Resources</strong></td></tr>
<tr><td>

- [nanoGPT GitHub repo](https://github.com/karpathy/nanoGPT)

</td></tr>
</table>

**Code:** [Book](week1/day03_transformers/README.md) | [mini_gpt.py](week1/day03_transformers/mini_gpt.py) | [visualizations.py](week1/day03_transformers/visualizations.py)

---

### [Day 4: Tokenization & Hugging Face Basics](week1/day04_tokenization_huggingface/)

<table>
<tr><td><strong>Morning (4 hrs) - Learn</strong></td></tr>
<tr><td>

- Karpathy - [Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE)
- [Hugging Face LLM Course - Chapter 1](https://huggingface.co/learn/llm-course/chapter1/1)

</td></tr>
<tr><td><strong>Afternoon (5 hrs) - Build</strong></td></tr>
<tr><td>

- Build a BPE tokenizer from scratch
- Use Hugging Face `transformers` library to load pre-trained models (GPT-2)
- Generate text, experiment with temperature and top-k/top-p sampling
- Compare outputs: GPT-2, DistilGPT-2, GPT-2-medium

</td></tr>
</table>

```bash
pip install transformers datasets accelerate tiktoken
```

**Code:** [Book](week1/day04_tokenization_huggingface/README.md) | [bpe_tokenizer.py](week1/day04_tokenization_huggingface/bpe_tokenizer.py) | [huggingface_generate.py](week1/day04_tokenization_huggingface/huggingface_generate.py) | [visualizations.py](week1/day04_tokenization_huggingface/visualizations.py)

---

### [Day 5: Prompt Engineering & LLM APIs](week1/day05_prompt_engineering_llm_apis/)

<table>
<tr><td><strong>Morning (4 hrs) - Learn</strong></td></tr>
<tr><td>

- DeepLearning.AI - [ChatGPT Prompt Engineering for Developers](https://www.deeplearning.ai/short-courses/chatgpt-prompt-engineering-for-developers/) (free, ~1.5 hrs)
- DeepLearning.AI - [Building Systems with the ChatGPT API](https://www.deeplearning.ai/short-courses/building-systems-with-chatgpt/) (free, ~1.5 hrs)

</td></tr>
<tr><td><strong>Afternoon (6 hrs) - Build</strong></td></tr>
<tr><td>

**Project 1: AI-Powered Resume Analyzer** - A Python CLI tool that takes a resume as input, extracts skills/experience/education, scores against a job description, suggests improvements, and generates a cover letter draft.

</td></tr>
</table>

> **Free API keys needed:** [Google Gemini](https://aistudio.google.com/) (60 req/min) | [Groq](https://console.groq.com/) (generous free tier) | [OpenAI](https://platform.openai.com/) ($5 credit)

---

### [Day 6: Embeddings & Vector Databases](week1/day06_embeddings_vector_databases/)

<table>
<tr><td><strong>Morning (4 hrs) - Learn</strong></td></tr>
<tr><td>

- DeepLearning.AI - [Building Applications with Vector Databases](https://www.deeplearning.ai/short-courses/building-applications-vector-databases/) (free)
- Vicki Boykis - [What are Embeddings?](https://vickiboykis.com/what_are_embeddings/)

</td></tr>
<tr><td><strong>Afternoon (5 hrs) - Build</strong></td></tr>
<tr><td>

**Project 2: Semantic Search Engine for Your Notes** - Collect 50+ text files, generate embeddings with `sentence-transformers/all-MiniLM-L6-v2` (runs locally), store in ChromaDB, build a CLI for semantic search with cross-encoder re-ranking.

</td></tr>
</table>

```bash
pip install chromadb sentence-transformers
```

---

### [Day 7: Review & Consolidation](week1/day07_review_consolidation/)

<table>
<tr><td><strong>Morning (4 hrs)</strong></td></tr>
<tr><td>

- Review all code from the week
- Revisit any weak concepts
- Read: [Hugging Face LLM Course Chapters 2-3](https://huggingface.co/learn/llm-course/chapter2/1)

</td></tr>
<tr><td><strong>Afternoon (5 hrs)</strong></td></tr>
<tr><td>

- Clean up Project 1 and Project 2
- Add proper error handling, README files, push to GitHub
- Write a short blog post / notes about what you learned

</td></tr>
</table>

---

## Week 2: Core GenAI Skills - RAG, Agents & Chains

> Learn the bread-and-butter of a Gen AI Engineer: building applications that combine LLMs with external data and tools.

---

### [Day 8: LangChain Fundamentals](week2/day08_langchain_fundamentals/)

<table>
<tr><td><strong>Morning (5 hrs) - Learn</strong></td></tr>
<tr><td>

- [LangChain Official Tutorials](https://python.langchain.com/docs/tutorials/):
  - Build a Simple LLM Application
  - Build a Chatbot
  - Build a RAG app

</td></tr>
<tr><td><strong>Afternoon (5 hrs) - Build</strong></td></tr>
<tr><td>

- Set up a LangChain project from scratch
- Build chains: simple -> sequential -> chain with memory
- Experiment with different LLMs (OpenAI, Groq/Llama, Gemini)
- Add conversation memory (BufferMemory, SummaryMemory)

</td></tr>
</table>

```bash
pip install langchain langchain-openai langchain-community
```

---

### [Day 9-10: RAG Deep Dive](week2/day09_rag_basics/)

<table>
<tr><td><strong>Day 9 - Learn & Build Basic RAG (10 hrs)</strong></td></tr>
<tr><td>

- DeepLearning.AI - [LangChain: Chat with Your Data](https://www.deeplearning.ai/short-courses/langchain-chat-with-your-data/) (free)
- Hugging Face Blog - [Code a simple RAG from scratch](https://huggingface.co/blog/ngxson/make-your-own-rag)
- [LangChain RAG Tutorial](https://python.langchain.com/docs/tutorials/rag/)

</td></tr>
<tr><td><strong>Day 10 - Build Advanced RAG (10 hrs)</strong></td></tr>
<tr><td>

**[Project 3: Chat With Any YouTube Channel](week2/day10_rag_advanced_youtube_project/)** - Download transcripts of last 20 videos, chunk intelligently (256, 512, 1024), embed and store in ChromaDB, build Q&A with source citations (video + timestamp), Streamlit UI.

</td></tr>
</table>

**Key RAG concepts to implement:**

| Concept | Description |
|:--------|:------------|
| Document Loading | Text splitting strategies for different content types |
| Embedding Models | OpenAI embeddings vs local sentence-transformers |
| Vector Store | ChromaDB for storage and retrieval |
| Retrieval Strategies | Similarity search, MMR (Maximum Marginal Relevance) |
| Prompt Templates | QA with context injection |

```bash
pip install youtube-transcript-api streamlit chromadb langchain
```

---

### [Day 11: AI Agents - Tool Use & ReAct Pattern](week2/day11_ai_agents_react_pattern/)

<table>
<tr><td><strong>Morning (5 hrs) - Learn</strong></td></tr>
<tr><td>

- DeepLearning.AI - [Functions, Tools and Agents with LangChain](https://www.deeplearning.ai/short-courses/functions-tools-agents-langchain/) (free)
- Microsoft - [AI Agents for Beginners, Lessons 1-4](https://github.com/microsoft/ai-agents-for-beginners)
- [ReAct Pattern](https://www.promptingguide.ai/techniques/react)

</td></tr>
<tr><td><strong>Afternoon (5 hrs) - Build</strong></td></tr>
<tr><td>

- Build an agent with LangChain that can search the web, do math, read/write files, and execute Python code
- Understand the **Thought -> Action -> Observation** loop

</td></tr>
</table>

```bash
pip install langchain-community duckduckgo-search
```

---

### [Day 12: LangGraph & Multi-Step Agents](week2/day12_langgraph_multi_step_agents/)

<table>
<tr><td><strong>Morning (4 hrs) - Learn</strong></td></tr>
<tr><td>

- [LangChain Academy - Intro to LangGraph](https://academy.langchain.com/courses/intro-to-langgraph) (free)
- [LangGraph Tutorials](https://langchain-ai.github.io/langgraph/tutorials/)

</td></tr>
<tr><td><strong>Afternoon (6 hrs) - Build</strong></td></tr>
<tr><td>

**Project 4: AI Research Agent** - An agent using LangGraph that generates search queries, searches the web, summarizes results, synthesizes a research report, and identifies follow-up questions. Graph-based workflow with conditional edges.

</td></tr>
</table>

```bash
pip install langgraph langchain-openai tavily-python
```

---

### [Day 13: Fine-Tuning LLMs](week2/day13_fine_tuning_llms/)

<table>
<tr><td><strong>Morning (5 hrs) - Learn</strong></td></tr>
<tr><td>

- DeepLearning.AI - [Finetuning Large Language Models](https://www.deeplearning.ai/short-courses/finetuning-large-language-models/) (free)
- [Hugging Face LLM Course - Chapter 11](https://huggingface.co/learn/llm-course/chapter11/1)
- [LoRA and QLoRA Guide](https://huggingface.co/docs/peft/conceptual_guides/lora)

</td></tr>
<tr><td><strong>Afternoon (5 hrs) - Build</strong></td></tr>
<tr><td>

**Project 5: Fine-Tune a Model for Your Use Case** - Pick a task (sentiment analysis, code explanation, domain Q&A), apply LoRA adapters to a small model (`phi-2` or `TinyLlama`), evaluate before vs after. Use Google Colab free tier if no GPU.

</td></tr>
</table>

```bash
pip install peft transformers datasets accelerate bitsandbytes
```

---

### [Day 14: Review, Refine & Integrate](week2/day14_review_refine_integrate/)

<table>
<tr><td><strong>Morning (4 hrs)</strong></td></tr>
<tr><td>

- Review Week 2 projects
- Try different chunking strategies, compare retrieval methods
- Read: [NirDiamant's GenAI Agents](https://github.com/NirDiamant/GenAI_Agents) for more patterns

</td></tr>
<tr><td><strong>Afternoon (5 hrs)</strong></td></tr>
<tr><td>

- Improve Project 3 (RAG): hybrid search, query rewriting, simple evaluation with 10 Q&A pairs
- Push all projects to GitHub with good READMEs

</td></tr>
</table>

---

## Week 3: Production Skills, Advanced Projects & Portfolio

> Build production-grade projects and learn deployment. This is what separates a learner from a hirable engineer.

---

### [Day 15: Multi-Agent Systems](week3/day15_multi_agent_systems/)

<table>
<tr><td><strong>Morning (4 hrs) - Learn</strong></td></tr>
<tr><td>

- DeepLearning.AI - [Multi AI Agent Systems with crewAI](https://www.deeplearning.ai/short-courses/multi-ai-agent-systems-with-crewai/) (free)
- Browse: [500 AI Agent Projects](https://github.com/ashishpatel26/500-AI-Agents-Projects) for ideas

</td></tr>
<tr><td><strong>Afternoon (6 hrs) - Build</strong></td></tr>
<tr><td>

**Project 6: AI Content Creation Pipeline** - Multi-agent system using CrewAI: Researcher (gathers facts), Writer (drafts blog post), Editor (reviews & improves), SEO Optimizer (keywords, meta descriptions). Output: publish-ready blog post.

</td></tr>
</table>

```bash
pip install crewai crewai-tools
```

---

### [Day 16: Agentic RAG & Advanced Patterns](week3/day16_agentic_rag/)

<table>
<tr><td><strong>Morning (4 hrs) - Learn</strong></td></tr>
<tr><td>

- DeepLearning.AI - [Building Agentic RAG with LlamaIndex](https://www.deeplearning.ai/short-courses/building-agentic-rag-with-llamaindex/) (free)
- [DataCamp Agentic RAG Tutorial](https://www.datacamp.com/tutorial/agentic-rag-tutorial)

</td></tr>
<tr><td><strong>Afternoon (6 hrs) - Build</strong></td></tr>
<tr><td>

**Project 7: AI Coding Assistant** - Agentic RAG that indexes a GitHub repo's codebase, answers questions about it, navigates between files following imports, and uses an agent loop to reformulate queries when first retrieval isn't sufficient.

</td></tr>
</table>

```bash
pip install llama-index llama-index-vector-stores-chroma
```

---

### [Day 17: Deployment & APIs](week3/day17_deployment_apis/)

<table>
<tr><td><strong>Morning (3 hrs) - Learn</strong></td></tr>
<tr><td>

- [FastAPI Tutorial](https://fastapi.tiangolo.com/tutorial/)
- [Streamlit Docs](https://docs.streamlit.io/get-started)
- [Docker Get Started](https://docs.docker.com/get-started/)

</td></tr>
<tr><td><strong>Afternoon (7 hrs) - Build</strong></td></tr>
<tr><td>

**Project 8: Deploy Your Best RAG App as a Full Product** - FastAPI backend (`POST /upload`, `POST /ask`, `GET /history`), Streamlit frontend, SQLite for conversation memory, streaming responses, Docker, deploy on Streamlit Cloud / HF Spaces / Railway.

</td></tr>
</table>

---

### [Day 18: Evaluation & Observability](week3/day18_evaluation_observability/)

<table>
<tr><td><strong>Morning (4 hrs) - Learn</strong></td></tr>
<tr><td>

- DeepLearning.AI - [Building and Evaluating Advanced RAG](https://www.deeplearning.ai/short-courses/building-evaluating-advanced-rag/) (free)
- [RAGAS Documentation](https://docs.ragas.io/en/latest/)
- [DeepEval Documentation](https://docs.confident-ai.com/)

</td></tr>
<tr><td><strong>Afternoon (5 hrs) - Build</strong></td></tr>
<tr><td>

- Create a test dataset of 20+ question-answer pairs
- Measure: faithfulness, answer relevancy, context precision, context recall
- Use RAGAS framework, generate evaluation reports
- Iterate: change chunk size, embedding model, prompt template and re-evaluate

</td></tr>
</table>

```bash
pip install ragas deepeval
```

---

### [Day 19: Multimodal AI & Vision](week3/day19_multimodal_ai_vision/)

<table>
<tr><td><strong>Morning (4 hrs) - Learn</strong></td></tr>
<tr><td>

- DeepLearning.AI - [Prompt Engineering for Vision Models](https://www.deeplearning.ai/short-courses/prompt-engineering-for-vision-models/) (free)
- Explore: GPT-4o (vision API), LLaVA (open source), Google Gemini (free multimodal API)

</td></tr>
<tr><td><strong>Afternoon (6 hrs) - Build</strong></td></tr>
<tr><td>

**Project 9: AI Invoice/Receipt Processor** - Upload photos of invoices/receipts, extract vendor name, date, total, line items, tax using a vision model, store in SQLite, generate monthly expense reports with charts, Streamlit UI.

</td></tr>
</table>

---

### [Day 20: Ethics, Safety & Responsible AI](week3/day20_ethics_safety/)

<table>
<tr><td><strong>Morning (3 hrs) - Learn</strong></td></tr>
<tr><td>

- [Generative AI for Everyone by Andrew Ng](https://www.coursera.org/learn/generative-ai-for-everyone) (free on Coursera, focus Weeks 2-3)
- Topics: hallucination mitigation, prompt injection attacks & defenses, bias in LLMs, EU AI Act basics

</td></tr>
<tr><td><strong>Afternoon (5 hrs) - Build</strong></td></tr>
<tr><td>

- Add safety features: input validation, prompt injection detection, output guardrails, content filters
- Red-team your own chatbot: try to make it produce incorrect outputs

</td></tr>
</table>

---

### [Day 21: Portfolio Day - Final Polish](week3/day21_portfolio_polish/)

<table>
<tr><td><strong>Morning (5 hrs)</strong></td></tr>
<tr><td>

- Pick your **3 best projects** and polish them
- Clean code, proper docstrings, comprehensive READMEs with screenshots/GIFs
- Architecture diagrams, tech stack documentation

</td></tr>
<tr><td><strong>Afternoon (5 hrs)</strong></td></tr>
<tr><td>

- Create a portfolio page (GitHub profile README or simple site)
- Write a LinkedIn post about your 3-week journey
- Plan continued learning: communities, people to follow, open source contributions

</td></tr>
</table>

---

## 📚 Quick Reference: All Free Resources

### Video Courses

| Resource | Topics | Duration |
|:---------|:-------|:---------|
| [Karpathy - Zero to Hero](https://karpathy.ai/zero-to-hero.html) | Neural nets, GPT from scratch | ~20 hrs |
| [DeepLearning.AI Short Courses](https://learn.deeplearning.ai/) | Prompt eng, RAG, agents, fine-tuning | ~1-2 hrs each |
| [Hugging Face LLM Course](https://huggingface.co/learn/llm-course/chapter1/1) | Transformers, NLP, fine-tuning | Self-paced |
| [LangChain Academy](https://academy.langchain.com/) | LangGraph, agents | Self-paced |
| [Microsoft AI Agents for Beginners](https://github.com/microsoft/ai-agents-for-beginners) | 12 lessons on AI agents | Self-paced |
| [CS50 AI (Harvard)](https://cs50.harvard.edu/ai/) | AI foundations | ~12 weeks |

### Documentation

| Resource | What You'll Find |
|:---------|:-----------------|
| [LangChain Docs & Tutorials](https://python.langchain.com/docs/tutorials/) | Chains, agents, RAG patterns |
| [LangGraph Tutorials](https://langchain-ai.github.io/langgraph/tutorials/) | Graph-based agent workflows |
| [Hugging Face Docs](https://huggingface.co/docs) | Transformers, datasets, model hub |
| [OpenAI Cookbook](https://cookbook.openai.com/) | API recipes and best practices |
| [Anthropic Claude Docs](https://docs.anthropic.com/) | Claude API, prompt engineering |
| [LlamaIndex Docs](https://docs.llamaindex.ai/) | Data framework for LLM apps |

### GitHub Repos

| Repo | What It's For |
|:-----|:--------------|
| [NirDiamant/GenAI_Agents](https://github.com/NirDiamant/GenAI_Agents) | Agent patterns & tutorials |
| [microsoft/ai-agents-for-beginners](https://github.com/microsoft/ai-agents-for-beginners) | Structured agent lessons |
| [ashishpatel26/500-AI-Agents-Projects](https://github.com/ashishpatel26/500-AI-Agents-Projects) | Project ideas & inspiration |
| [langchain-ai/agents-from-scratch](https://github.com/langchain-ai/agents-from-scratch) | Build agents step by step |

### Free API Keys

| Provider | Free Tier | Get Started |
|:---------|:----------|:------------|
| Google Gemini | 60 req/min, very generous | [aistudio.google.com](https://aistudio.google.com/) |
| Groq | Fast inference, generous free tier | [console.groq.com](https://console.groq.com/) |
| OpenAI | $5 credit for new accounts | [platform.openai.com](https://platform.openai.com/) |
| Hugging Face | Free inference API | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |
| Cohere | 1,000 req/month free | [dashboard.cohere.com](https://dashboard.cohere.com/) |

### Free Compute (for fine-tuning)

| Platform | GPU Available | Get Started |
|:---------|:-------------|:------------|
| Google Colab | T4 (free tier) | [colab.research.google.com](https://colab.research.google.com/) |
| Kaggle Notebooks | T4/P100, 30 hrs/week | [kaggle.com/code](https://www.kaggle.com/code) |
| Lightning.ai | Free GPU credits | [lightning.ai](https://lightning.ai/) |

---

## 🏗 Project Summary

| # | Project | Day | Key Skills | Difficulty |
|:-:|:--------|:---:|:-----------|:----------:|
| 1 | **AI Resume Analyzer** | 5 | LLM APIs, prompt engineering, structured output | ⭐⭐ |
| 2 | **Semantic Search Engine** | 6 | Embeddings, vector DB, sentence-transformers | ⭐⭐ |
| 3 | **Chat With YouTube Channel** | 9-10 | RAG, chunking, ChromaDB, Streamlit | ⭐⭐⭐ |
| 4 | **AI Research Agent** | 12 | LangGraph, web search, multi-step reasoning | ⭐⭐⭐ |
| 5 | **Fine-Tuned Domain Model** | 13 | Hugging Face, LoRA, PEFT, evaluation | ⭐⭐⭐ |
| 6 | **AI Content Pipeline** | 15 | Multi-agent, CrewAI, tool use | ⭐⭐⭐⭐ |
| 7 | **AI Coding Assistant** | 16 | Agentic RAG, code understanding | ⭐⭐⭐⭐ |
| 8 | **Deployed RAG Product** | 17 | FastAPI, Docker, Streamlit, deployment | ⭐⭐⭐⭐ |
| 9 | **Invoice/Receipt Processor** | 19 | Multimodal AI, vision, structured extraction | ⭐⭐⭐⭐ |

---

## Setup

### Environment

```bash
# Install Miniconda (if not installed)
# https://docs.conda.io/en/latest/miniconda.html

# Create and activate environment
conda create -n genai python=3.11 -y
conda activate genai

# Install core packages
conda install pytorch matplotlib numpy scikit-learn -y
pip install transformers datasets langchain chromadb sentence-transformers
```

### Repository Structure

```
gen_ai_engineer/
├── README.md                                    # This file
├── week1/
│   ├── day01_neural_networks_backpropagation/
│   │   ├── README.md                            # Book: Neural Networks & Backprop
│   │   ├── micrograd.py                         # Autograd engine + MLP
│   │   ├── exercises.py                         # Hands-on exercises
│   │   ├── visualizations.py                    # Generate 8 concept diagrams
│   │   └── viz_*.png                            # Pre-generated diagrams
│   ├── day02_language_modeling_nlp/
│   │   ├── README.md                            # Book: Language Modeling & NLP
│   │   ├── bigram.py                            # Counting-based bigram model
│   │   ├── bigram_neural.py                     # Neural network bigram
│   │   ├── mlp_lm.py                            # MLP language model
│   │   └── names.txt                            # Dataset (32K names)
│   ├── day03_transformers/
│   │   ├── README.md                            # Book: Transformers
│   │   ├── mini_gpt.py                          # Decoder-only Transformer LM
│   │   ├── visualizations.py                    # Generate 8 transformer diagrams
│   │   └── viz_*.png                            # Pre-generated diagrams
│   ├── day04_tokenization_huggingface/
│   │   ├── README.md                            # Book: Tokenization & HF
│   │   ├── bpe_tokenizer.py                     # BPE tokenizer from scratch
│   │   ├── huggingface_generate.py              # GPT-2 loading & generation
│   │   ├── visualizations.py                    # Generate 8 concept diagrams
│   │   └── viz_*.png                            # Pre-generated diagrams
│   └── day05-07 .../
├── week2/
│   └── day08-14 .../
└── week3/
    └── day15-21 .../
```

---

<div align="center">

> **The goal is not to finish everything perfectly. It's to build muscle memory for the GenAI stack.**
> A working ugly project teaches more than a polished tutorial you followed.
> When stuck, read the error, check the docs, ask an LLM - that IS the job of a GenAI engineer.

</div>
