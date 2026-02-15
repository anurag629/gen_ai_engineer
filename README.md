# Gen AI Engineer - 3-Week Intensive Learning Plan

> **Prerequisites you already have:** Python, Pandas, Matplotlib, basic data science
> **Time commitment:** Full day (~10-12 hours/day)
> **Approach:** Learn a concept in the morning, build a project in the afternoon/evening

---

## WEEK 1: Foundations - Deep Learning, Transformers & LLMs

The goal this week is to understand how neural networks and LLMs actually work under the hood. Without this, everything else is just API calls with no understanding.

---

### Day 1: Neural Networks & Backpropagation

**Learn (Morning - 4 hrs)**
- Andrej Karpathy - "The spelled-out intro to neural networks and backpropagation: building micrograd"
  - YouTube: https://www.youtube.com/watch?v=VMj-3S1tku0
- Follow along and code micrograd yourself

**Build (Afternoon - 4 hrs)**
- Rebuild micrograd from scratch without looking at the video
- Extend it: add tanh, relu, sigmoid activation functions
- Train a small classifier on a toy dataset

**Resources:**
- GitHub repo: https://github.com/karpathy/micrograd

---

### Day 2: Language Modeling & NLP Basics

**Learn (Morning - 5 hrs)**
- Karpathy - "The spelled-out intro to language modeling: building makemore"
  - YouTube: https://www.youtube.com/watch?v=PaCmpygFfXo
- Karpathy - "Building makemore Part 2: MLP"
  - YouTube: https://www.youtube.com/watch?v=TCH_1BHY58I

**Build (Afternoon - 4 hrs)**
- Build a character-level name generator from scratch
- Train it on a dataset of Indian names (download from Kaggle)
- Experiment: change hidden layer sizes, learning rates, plot loss curves with matplotlib

**Resources:**
- GitHub: https://github.com/karpathy/makemore

---

### Day 3: Transformers - The Core of All GenAI

**Learn (Morning - 5 hrs)**
- Karpathy - "Let's build GPT: from scratch, in code, spelled out"
  - YouTube: https://www.youtube.com/watch?v=kCc8FmEb1nY
- Read the blog: "The Illustrated Transformer" by Jay Alammar
  - https://jalammar.github.io/illustrated-transformer/

**Build (Afternoon - 5 hrs)**
- Code a mini GPT from scratch following Karpathy's video
- Train it on a text file of your choice (Shakespeare, song lyrics, etc.)
- Generate text and observe how quality changes with training

**Resources:**
- GitHub: https://github.com/karpathy/nanoGPT

---

### Day 4: Tokenization & Hugging Face Basics

**Learn (Morning - 4 hrs)**
- Karpathy - "Let's build the GPT Tokenizer"
  - YouTube: https://www.youtube.com/watch?v=zduSFxRajkE
- Hugging Face LLM Course - Chapter 1: Introduction to Transformers
  - https://huggingface.co/learn/llm-course/chapter1/1

**Build (Afternoon - 5 hrs)**
- Build a BPE tokenizer from scratch
- Then use Hugging Face `transformers` library:
  ```
  pip install transformers datasets accelerate
  ```
- Load a pre-trained model (GPT-2), generate text, experiment with temperature and top-k/top-p sampling
- Compare outputs of different models: GPT-2, DistilGPT-2, GPT-2-medium

---

### Day 5: Prompt Engineering & LLM APIs

**Learn (Morning - 4 hrs)**
- DeepLearning.AI - "ChatGPT Prompt Engineering for Developers" (free, ~1.5 hrs)
  - https://www.deeplearning.ai/short-courses/chatgpt-prompt-engineering-for-developers/
- DeepLearning.AI - "Building Systems with the ChatGPT API" (free, ~1.5 hrs)
  - https://www.deeplearning.ai/short-courses/building-systems-with-chatgpt/

**Build (Afternoon - 6 hrs)**

**PROJECT 1: AI-Powered Resume Analyzer**
Build a Python CLI tool that:
- Takes a resume (PDF/text) as input
- Uses OpenAI/Anthropic API to:
  - Extract skills, experience, education
  - Score the resume against a job description
  - Suggest improvements
  - Generate a cover letter draft
- Save results as a structured JSON + markdown report

```
You'll need (free tiers available):
- OpenAI API key (free $5 credit for new accounts) OR
- Google Gemini API key (free tier: 60 requests/min) OR
- Groq API key (free tier: very generous)
```

---

### Day 6: Embeddings & Vector Databases

**Learn (Morning - 4 hrs)**
- DeepLearning.AI - "Building Applications with Vector Databases" (free)
  - https://www.deeplearning.ai/short-courses/building-applications-vector-databases/
- Read: "What are Embeddings?" - Vicki Boykis
  - https://vickiboykis.com/what_are_embeddings/

**Build (Afternoon - 5 hrs)**

**PROJECT 2: Semantic Search Engine for Your Notes**
Build a local semantic search engine:
- Collect 50+ text files/notes/articles (or scrape Wikipedia articles)
- Generate embeddings using a free model:
  - `sentence-transformers/all-MiniLM-L6-v2` (runs locally, no API needed)
- Store in ChromaDB (free, local vector database)
- Build a CLI where you type a question and get the most relevant documents
- Add a re-ranking step using cross-encoder

```
pip install chromadb sentence-transformers
```

---

### Day 7: Review & Consolidation

**Morning (4 hrs)**
- Review all code you wrote this week
- Revisit any concepts that felt weak
- Read: Hugging Face LLM Course Chapters 2-3
  - https://huggingface.co/learn/llm-course/chapter2/1

**Afternoon (5 hrs)**
- Clean up Project 1 and Project 2
- Add proper error handling, README files, push to GitHub
- Write a short blog post / notes about what you learned (for your own reference)

---

## WEEK 2: Core GenAI Skills - RAG, Agents & Chains

This week you learn the bread-and-butter of a Gen AI Engineer: building applications that combine LLMs with external data and tools.

---

### Day 8: LangChain Fundamentals

**Learn (Morning - 5 hrs)**
- LangChain official tutorials (free):
  - https://python.langchain.com/docs/tutorials/
  - Complete: "Build a Simple LLM Application"
  - Complete: "Build a Chatbot"
  - Complete: "Build a Retrieval Augmented Generation (RAG) app"

**Build (Afternoon - 5 hrs)**
- Set up a LangChain project from scratch
- Build chains: simple chain -> sequential chain -> chain with memory
- Experiment with different LLMs through LangChain (OpenAI, Groq/Llama, Gemini)
- Add conversation memory (BufferMemory, SummaryMemory)

```
pip install langchain langchain-openai langchain-community
```

---

### Day 9-10: RAG (Retrieval Augmented Generation) - Deep Dive

**Day 9 - Learn & Build Basic RAG (10 hrs)**

- DeepLearning.AI - "LangChain: Chat with Your Data" (free)
  - https://www.deeplearning.ai/short-courses/langchain-chat-with-your-data/
- Hugging Face Blog - "Code a simple RAG from scratch"
  - https://huggingface.co/blog/ngxson/make-your-own-rag
- LangChain RAG tutorial:
  - https://python.langchain.com/docs/tutorials/rag/

**Day 10 - Build Advanced RAG (10 hrs)**

**PROJECT 3: Chat With Any YouTube Channel**
Build a full RAG application that:
- Takes a YouTube channel URL
- Downloads transcripts of the last 20 videos (use `youtube-transcript-api`)
- Chunks transcripts intelligently (experiment with chunk sizes: 256, 512, 1024)
- Embeds and stores in ChromaDB
- Lets you ask questions like:
  - "What does this creator think about Python vs JavaScript?"
  - "Summarize their views on AI"
  - "What topics do they cover most?"
- Add source citations (which video + timestamp)
- Build a simple Streamlit UI

```
pip install youtube-transcript-api streamlit chromadb langchain
```

**Key RAG concepts to implement:**
- Document loading & text splitting strategies
- Embedding models (try both OpenAI and local sentence-transformers)
- Vector store (ChromaDB)
- Retrieval strategies: similarity search, MMR (Maximum Marginal Relevance)
- Prompt templates for QA with context

---

### Day 11: AI Agents - Tool Use & ReAct Pattern

**Learn (Morning - 5 hrs)**
- DeepLearning.AI - "Functions, Tools and Agents with LangChain" (free)
  - https://www.deeplearning.ai/short-courses/functions-tools-agents-langchain/
- Microsoft's "AI Agents for Beginners" - Lessons 1-4 (free)
  - https://github.com/microsoft/ai-agents-for-beginners
- Read about the ReAct pattern:
  - https://www.promptingguide.ai/techniques/react

**Build (Afternoon - 5 hrs)**
- Build a simple agent with LangChain that can:
  - Search the web (using Tavily or DuckDuckGo search)
  - Do math calculations
  - Read/write files
  - Execute Python code
- Understand the Thought -> Action -> Observation loop

```
pip install langchain-community duckduckgo-search
```

---

### Day 12: LangGraph & Multi-Step Agents

**Learn (Morning - 4 hrs)**
- LangChain Academy - "Introduction to LangGraph" (free)
  - https://academy.langchain.com/courses/intro-to-langgraph
- LangGraph documentation + tutorials:
  - https://langchain-ai.github.io/langgraph/tutorials/

**Build (Afternoon - 6 hrs)**

**PROJECT 4: AI Research Agent**
Build an agent using LangGraph that:
- Takes a research topic as input
- Step 1: Generates 5 search queries about the topic
- Step 2: Searches the web for each query (DuckDuckGo)
- Step 3: Reads and summarizes each result
- Step 4: Synthesizes all summaries into a coherent research report
- Step 5: Identifies gaps and generates follow-up questions
- Has a graph-based workflow with conditional edges (if results are poor, retry with different queries)
- Outputs a structured markdown report

```
pip install langgraph langchain-openai tavily-python
```

---

### Day 13: Fine-Tuning LLMs

**Learn (Morning - 5 hrs)**
- DeepLearning.AI - "Finetuning Large Language Models" (free)
  - https://www.deeplearning.ai/short-courses/finetuning-large-language-models/
- Hugging Face LLM Course - Chapter 11: Fine-tuning LLMs
  - https://huggingface.co/learn/llm-course/chapter11/1
- Read about LoRA and QLoRA:
  - https://huggingface.co/docs/peft/conceptual_guides/lora

**Build (Afternoon - 5 hrs)**

**PROJECT 5: Fine-Tune a Model for Your Use Case**
- Pick a task: sentiment analysis on product reviews, code explanation, or Q&A on a specific domain
- Use Hugging Face + PEFT (Parameter Efficient Fine-Tuning):
  - Load a small model (e.g., `microsoft/phi-2` or `TinyLlama/TinyLlama-1.1B`)
  - Apply LoRA adapters
  - Fine-tune on a dataset from Hugging Face Hub
  - Evaluate before vs after fine-tuning
- If you don't have a GPU, use Google Colab free tier (T4 GPU)

```
pip install peft transformers datasets accelerate bitsandbytes
```

---

### Day 14: Review, Refine & Integrate

**Morning (4 hrs)**
- Review Week 2 projects
- Revisit RAG concepts: try different chunking strategies, compare retrieval methods
- Read: NirDiamant's GenAI Agents repo for more patterns
  - https://github.com/NirDiamant/GenAI_Agents

**Afternoon (5 hrs)**
- Improve Project 3 (RAG) with:
  - Hybrid search (keyword + semantic)
  - Query rewriting (let the LLM rephrase the user's question for better retrieval)
  - Add a simple evaluation: create 10 Q&A pairs and measure answer quality
- Push all projects to GitHub with good READMEs

---

## WEEK 3: Production Skills, Advanced Projects & Portfolio

This week you build production-grade projects and learn deployment. This is what separates a learner from a hirable engineer.

---

### Day 15: Multi-Agent Systems

**Learn (Morning - 4 hrs)**
- DeepLearning.AI - "Multi AI Agent Systems with crewAI" (free)
  - https://www.deeplearning.ai/short-courses/multi-ai-agent-systems-with-crewai/
- Browse: 500 AI Agent Projects for ideas
  - https://github.com/ashishpatel26/500-AI-Agents-Projects

**Build (Afternoon - 6 hrs)**

**PROJECT 6: AI Content Creation Pipeline (Multi-Agent)**
Build a multi-agent system using CrewAI or LangGraph:
- **Agent 1 - Researcher:** Takes a topic, searches the web, gathers facts
- **Agent 2 - Writer:** Takes research, writes a blog post draft
- **Agent 3 - Editor:** Reviews the draft, checks facts, improves quality
- **Agent 4 - SEO Optimizer:** Adds keywords, meta description, suggests title variants
- Output: A publish-ready blog post with sources

```
pip install crewai crewai-tools
```

---

### Day 16: Agentic RAG & Advanced Patterns

**Learn (Morning - 4 hrs)**
- DeepLearning.AI - "Building Agentic RAG with LlamaIndex" (free)
  - https://www.deeplearning.ai/short-courses/building-agentic-rag-with-llamaindex/
- DataCamp tutorial on Agentic RAG:
  - https://www.datacamp.com/tutorial/agentic-rag-tutorial

**Build (Afternoon - 6 hrs)**

**PROJECT 7: AI Coding Assistant (Agentic RAG)**
Build an agentic RAG system that:
- Indexes a GitHub repository's codebase (clone -> read files -> chunk -> embed)
- Answers questions about the code: "How does authentication work?", "Where is the database connected?"
- Can navigate between files following imports
- Uses an agent loop: if the first retrieval isn't enough, it reformulates and searches again
- Add tool: the agent can run `grep`-like searches on the codebase for exact matches

```
pip install llama-index llama-index-vector-stores-chroma
```

---

### Day 17: Deployment & APIs

**Learn (Morning - 3 hrs)**
- FastAPI crash course (free):
  - https://fastapi.tiangolo.com/tutorial/
- Streamlit docs (for rapid UI):
  - https://docs.streamlit.io/get-started
- Docker basics for ML (if not known):
  - https://docs.docker.com/get-started/

**Build (Afternoon - 7 hrs)**

**PROJECT 8: Deploy Your Best RAG App as a Full Product**
Take your best project (Project 3 or 7) and make it production-ready:
- Wrap it in a **FastAPI** backend with proper endpoints:
  - `POST /upload` - upload documents
  - `POST /ask` - ask a question
  - `GET /history` - get conversation history
- Build a **Streamlit** frontend
- Add:
  - Conversation memory (store in SQLite)
  - Streaming responses (Server-Sent Events)
  - Error handling and input validation
  - Rate limiting
- Dockerize the whole thing
- Deploy free on:
  - **Streamlit Cloud** (streamlit.io) OR
  - **Hugging Face Spaces** (huggingface.co/spaces) OR
  - **Railway.app** (free tier)

---

### Day 18: Evaluation & Observability

**Learn (Morning - 4 hrs)**
- DeepLearning.AI - "Building and Evaluating Advanced RAG" (free)
  - https://www.deeplearning.ai/short-courses/building-evaluating-advanced-rag/
- Learn about LLM evaluation frameworks:
  - RAGAS: https://docs.ragas.io/en/latest/
  - DeepEval: https://docs.confident-ai.com/

**Build (Afternoon - 5 hrs)**
- Add evaluation to your RAG project:
  - Create a test dataset of 20+ question-answer pairs
  - Measure: faithfulness, answer relevancy, context precision, context recall
  - Use RAGAS framework
  - Generate an evaluation report
  - Iterate: change chunk size, embedding model, prompt template and re-evaluate

```
pip install ragas deepeval
```

---

### Day 19: Multimodal AI & Vision

**Learn (Morning - 4 hrs)**
- DeepLearning.AI - "Prompt Engineering for Vision Models" (free)
  - https://www.deeplearning.ai/short-courses/prompt-engineering-for-vision-models/
- Explore multimodal models:
  - GPT-4o (vision capabilities via API)
  - LLaVA (open source, runs locally)
  - Google Gemini (free API, multimodal)

**Build (Afternoon - 6 hrs)**

**PROJECT 9: AI Invoice/Receipt Processor**
Build a multimodal AI application that:
- Takes photos of invoices/receipts (upload images)
- Uses a vision model to extract:
  - Vendor name, date, total amount, line items, tax
- Structures the data as JSON
- Stores in a SQLite database
- Generates monthly expense reports with charts (use matplotlib/plotly)
- Streamlit UI for uploading and viewing reports

---

### Day 20: Ethics, Safety & Responsible AI

**Learn (Morning - 3 hrs)**
- DeepLearning.AI - "Generative AI for Everyone" by Andrew Ng (free on Coursera)
  - https://www.coursera.org/learn/generative-ai-for-everyone
  - Focus on Weeks 2-3 covering limitations, risks, responsible use
- Read about:
  - Hallucination mitigation strategies
  - Prompt injection attacks and defenses
  - Bias in LLMs
  - EU AI Act basics

**Build (Afternoon - 5 hrs)**
- Add safety features to your projects:
  - Input validation / prompt injection detection
  - Output guardrails (check for harmful content)
  - Implement a simple content filter
- Experiment with red-teaming your own chatbot: try to make it say something wrong

---

### Day 21: Portfolio Day - Final Polish & Next Steps

**Morning (5 hrs)**
- Pick your **3 best projects** and polish them:
  - Clean code, proper docstrings for public functions
  - Comprehensive README with:
    - What it does (with screenshots/GIFs)
    - Architecture diagram (use draw.io or excalidraw)
    - How to run it
    - Tech stack
  - Push to GitHub with proper `.gitignore`, `requirements.txt`

**Afternoon (5 hrs)**
- Create a portfolio page (GitHub profile README or simple site)
- Write a LinkedIn post about your 3-week learning journey
- Plan your continued learning:
  - Join communities: r/LocalLLaMA, Hugging Face Discord, LangChain Discord
  - Follow: Andrej Karpathy, Simon Willison, Harrison Chase, Lilian Weng's blog
  - Contribute to open source: LangChain, LlamaIndex, or Hugging Face repos

---

## Quick Reference: All Free Resources

### Video Courses (Free)
| Resource | Link | Topics |
|---|---|---|
| Karpathy - Zero to Hero | https://karpathy.ai/zero-to-hero.html | Neural nets, GPT from scratch |
| DeepLearning.AI Short Courses | https://learn.deeplearning.ai/ | Prompt eng, RAG, agents, fine-tuning |
| Hugging Face LLM Course | https://huggingface.co/learn/llm-course/chapter1/1 | Transformers, NLP, fine-tuning |
| LangChain Academy | https://academy.langchain.com/ | LangGraph, agents |
| Microsoft AI Agents for Beginners | https://github.com/microsoft/ai-agents-for-beginners | 12 lessons on AI agents |
| CS50 AI (Harvard) | https://cs50.harvard.edu/ai/ | AI foundations |

### Documentation (Free)
| Resource | Link |
|---|---|
| LangChain Docs & Tutorials | https://python.langchain.com/docs/tutorials/ |
| LangGraph Tutorials | https://langchain-ai.github.io/langgraph/tutorials/ |
| Hugging Face Docs | https://huggingface.co/docs |
| OpenAI Cookbook | https://cookbook.openai.com/ |
| Anthropic Claude Docs | https://docs.anthropic.com/ |
| LlamaIndex Docs | https://docs.llamaindex.ai/ |

### GitHub Repos (Free)
| Repo | Link | Use |
|---|---|---|
| NirDiamant/GenAI_Agents | https://github.com/NirDiamant/GenAI_Agents | Agent patterns & tutorials |
| microsoft/ai-agents-for-beginners | https://github.com/microsoft/ai-agents-for-beginners | Structured lessons |
| ashishpatel26/500-AI-Agents-Projects | https://github.com/ashishpatel26/500-AI-Agents-Projects | Project ideas |
| langchain-ai/agents-from-scratch | https://github.com/langchain-ai/agents-from-scratch | Build agents step by step |

### Free API Keys (for projects)
| Provider | Free Tier | Link |
|---|---|---|
| Google Gemini | 60 req/min, very generous | https://aistudio.google.com/ |
| Groq | Fast inference, free tier | https://console.groq.com/ |
| OpenAI | $5 credit for new accounts | https://platform.openai.com/ |
| Hugging Face | Free inference API | https://huggingface.co/settings/tokens |
| Cohere | 1000 req/month free | https://dashboard.cohere.com/ |

### Free Compute (for fine-tuning)
| Platform | GPU | Link |
|---|---|---|
| Google Colab | T4 (free tier) | https://colab.research.google.com/ |
| Kaggle Notebooks | T4/P100, 30 hrs/week | https://www.kaggle.com/code |
| Lightning.ai | Free GPU credits | https://lightning.ai/ |

---

## Project Summary

| # | Project | Key Skills | Week |
|---|---|---|---|
| 1 | AI Resume Analyzer | LLM APIs, prompt engineering, structured output | 1 |
| 2 | Semantic Search Engine | Embeddings, vector DB, sentence-transformers | 1 |
| 3 | Chat With YouTube Channel | RAG, chunking, ChromaDB, Streamlit | 2 |
| 4 | AI Research Agent | LangGraph, web search, multi-step reasoning | 2 |
| 5 | Fine-Tuned Domain Model | Hugging Face, LoRA, PEFT, evaluation | 2 |
| 6 | AI Content Pipeline | Multi-agent, CrewAI, tool use | 3 |
| 7 | AI Coding Assistant | Agentic RAG, code understanding | 3 |
| 8 | Deployed RAG Product | FastAPI, Docker, Streamlit, deployment | 3 |
| 9 | Invoice/Receipt Processor | Multimodal AI, vision, structured extraction | 3 |

---

> **Remember:** The goal is not to finish everything perfectly. It's to build muscle memory
> for the GenAI stack. A working ugly project teaches more than a polished tutorial you followed.
> When stuck, read the error, check the docs, ask an LLM - that IS the job of a GenAI engineer.
