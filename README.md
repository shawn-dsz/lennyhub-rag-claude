# 🎙️ LennyHub RAG

A RAG (Retrieval-Augmented Generation) system built on transcripts from [Lenny's Podcast](https://www.lennysnewsletter.com/podcast), featuring conversations with top product leaders and growth experts.

## 🌟 Features

- **Knowledge Graph RAG**: Uses LightRAG for advanced retrieval with entity and relationship extraction
- **Multi-Modal Ready**: Built on RAG-Anything framework (supports text, images, tables, equations)
- **Hybrid Search**: Combines local context and global knowledge graph search
- **70+ Sample Questions**: Pre-built questions covering career advice, growth frameworks, and more
- **Interactive Query Mode**: Easy-to-use CLI for exploring the content

## 📚 Included Transcripts

### Ada Chen Rekhi
Executive coach and co-founder of Notejoy. Former SVP of Marketing at SurveyMonkey.

**Topics covered:**
- 🔄 Curiosity Loops - Structured approach to gathering advice
- 🎯 Values Exercises - Identifying personal values for career decisions
- 📈 Explore & Exploit Framework - Career development strategy
- 🥦 Eating Your Vegetables - Skill development through deliberate practice
- 💑 Co-founder Partnerships - Working with your spouse

### Adam Fishman
Former VP of Growth at Lyft and Patreon, CPO at Imperfect Foods.

**Topics covered:**
- 🎯 Growth Competency Model - Framework for hiring/evaluating growth talent
- 🚀 Onboarding Optimization - Impact of first-user experiences
- 🏢 PMF for Candidates - People, Mission, Financials framework for choosing companies
- 📊 Growth Strategy - From execution to influence
- 🔍 Opinionated Defaults - Smart defaults for user success

## 🚀 Quick Start

### Installation

```bash
# Clone or navigate to the project
cd lennyhub-rag

# Install dependencies
pip install -r requirements.txt

# Configure your OpenAI API key
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### Build the RAG

```bash
python build_transcript_rag.py
```

Expected: 2-3 minutes, ~$0.25 in API costs

### Query the System

```bash
# Interactive mode (recommended!)
python query_rag.py --interactive

# Single question
python query_rag.py "What is a curiosity loop?"

# Run examples
python query_rag.py
```

## 💡 Example Queries

### Career Strategy
```
"What is the explore and exploit framework for career development?"
"How do you avoid being the boiled frog in your career?"
"What advice does Ada give about early career strategy?"
```

### Growth & Product
```
"What are the four components of the growth competency model?"
"Why is onboarding important for growth?"
"How can onboarding improve retention?"
```

### Decision Making
```
"What is a curiosity loop and how does it work?"
"What is the PMF framework for choosing a company?"
"How should you use values to make career decisions?"
```

### Frameworks & Concepts
```
"What is the inner vs outer scorecard concept?"
"What are opinionated defaults?"
"What is the eating your vegetables concept?"
```

## 📖 Documentation

- **[INSTALL.md](INSTALL.md)** - Detailed installation and setup guide
- **[README_TRANSCRIPT_RAG.md](README_TRANSCRIPT_RAG.md)** - Technical documentation
- **[sample_questions.txt](sample_questions.txt)** - 70+ pre-written questions

## 🗂️ Project Structure

```
lennyhub-rag/
├── data/                      # Source transcripts
│   ├── Ada Chen Rekhi.txt
│   └── Adam Fishman.txt
├── rag_storage/              # Generated knowledge graph (after build)
├── build_transcript_rag.py   # Build the RAG system
├── query_rag.py              # Query interface
├── requirements.txt          # Python dependencies
├── sample_questions.txt      # 70+ example questions
├── .env                      # Your API keys (create from .env.example)
└── README.md                 # This file
```

## 🎯 Use Cases

- **Career Research**: Learn from top executives about career strategy
- **Interview Prep**: Study frameworks and concepts from industry leaders
- **Growth Learning**: Understand growth strategies from Lyft and Patreon
- **Decision Making**: Apply proven frameworks to your own decisions
- **Product Strategy**: Learn about onboarding, retention, and user experience

## 🧠 How It Works

1. **Document Processing**: Transcripts are parsed and chunked
2. **Entity Extraction**: Named entities and relationships are identified using GPT-4o-mini
3. **Knowledge Graph**: Entities and relationships form a queryable graph
4. **Embeddings**: Text is embedded using OpenAI's text-embedding-3-small
5. **Hybrid Search**: Queries use both vector similarity and graph traversal
6. **LLM Synthesis**: Results are synthesized into coherent answers

## 💰 Cost Breakdown

- **Initial Build**: ~$0.25
  - Text embeddings: ~$0.04
  - Entity extraction: ~$0.20
- **Per Query**: ~$0.001-0.01
- **Cached Queries**: Free

## 🔧 Requirements

- Python 3.9+
- OpenAI API key
- ~200MB disk space for rag_storage
- 2GB+ RAM recommended

## 🤝 Contributing

Have more Lenny's Podcast transcripts to add? Want to improve the queries? Contributions welcome!

## 📝 License

See [LICENSE](LICENSE) file for details.

## 🙏 Credits

- **Transcripts**: From [Lenny's Podcast](https://www.lennysnewsletter.com/podcast)
- **RAG Framework**: [RAG-Anything](https://github.com/HKUDS/RAG-Anything) by HKUDS
- **LightRAG**: [LightRAG](https://github.com/HKUDS/LightRAG) for knowledge graph RAG
- **Guests**:
  - [Ada Chen Rekhi](https://www.adachen.com/)
  - [Adam Fishman](https://fishmanafnewsletter.com/)

## 📧 Questions?

Review the [INSTALL.md](INSTALL.md) for troubleshooting or check [sample_questions.txt](sample_questions.txt) for query inspiration.

---

Built with ❤️ using RAG-Anything and LightRAG
