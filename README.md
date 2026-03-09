# Historical Music Theory Query System
### Multi-Database RAG with Author Filtering

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-red)](https://streamlit.io/)

## Overview

The **Historical Music Theory Query System** is a Streamlit application that allows you to query a database of music theory texts using a large language model (LLM) with retrieval-augmented generation (RAG). The system supports multiple databases and author-level filtering, enabling focused queries across a rich corpus of historical musicological sources in Latin, Italian, and English.

Learn more about the system and how to write effective prompts using the tools in the left-hand sidebar.

## Corpora

The system draws on several major digital repositories of historical music theory:

| Database | Description | Language |
|---|---|---|
| **TML** | [Thesaurus Musicarum Latinarum](https://chmtl.indiana.edu/tml/) | Latin |
| **TMI** | [Thesaurus Musicarum Italicarum](https://chmtl.indiana.edu/tmi/) | Italian |
| **TME** | [Thesaurus Musicarum Encyclopediarum](https://chmtl.indiana.edu/tme/) | Various |
| **TCP** | [Text Creation Partnership](https://textcreationpartnership.org/) | English |

Each corpus is parsed, chunked, and indexed into a [ChromaDB](https://www.trychroma.com/) vector store for semantic retrieval.

## Features

- 🎵 **RAG-powered queries** over a curated corpus of historical music theory texts
- 📚 **Multi-database support** — search across TML, TMI, TME, and TCP simultaneously or individually
- 🔍 **Author filtering** — narrow results to specific theorists or sources
- 💬 **LLM integration** — natural language question answering grounded in primary sources
- 🖥️ **Streamlit interface** — accessible, browser-based UI with sidebar navigation

## Repository Structure

```
theory_llm/
│
├── theory_llm_multi.py             # Main Streamlit app (multi-database)
├── theory_llm_5.py                 # Earlier Streamlit app versions
├── theory_llm_6.py
├── theory_llm_7.py
│
├── TCP_Parser_Chroma_Builder.ipynb # Corpus parsers and ChromaDB builders
├── TME_Parser_Chroma_Builder.ipynb
├── TMI_Parser_Chroma_Builder.ipynb
├── TML_Parser_Chroma_Builder.ipynb
│
├── get_chmtl_texts.ipynb           # Text retrieval notebooks
├── get_tmi_texts.ipynb
├── chroma-db-check.ipynb           # ChromaDB inspection and validation
│
├── english_html_metadata.csv       # Corpus metadata files
├── italian_tmi_metadata.csv
├── latin_tei_metadata.csv
│
├── chroma_compatibility_report.json
├── requirements.txt
└── .python-version
```

## Getting Started

### Prerequisites

- Python 3.9 or higher (see `.python-version` for the project-specified version)
- pip

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/RichardFreedman/theory_llm.git
   cd theory_llm
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure your environment variables (e.g., API key for your LLM provider):
   ```bash
   cp .env.example .env
   # Edit .env with your credentials
   ```

4. Build ChromaDB vector stores using the corpus parser notebooks:
   - `TML_Parser_Chroma_Builder.ipynb` — Latin texts
   - `TMI_Parser_Chroma_Builder.ipynb` — Italian texts
   - `TME_Parser_Chroma_Builder.ipynb` — Encyclopedia texts
   - `TCP_Parser_Chroma_Builder.ipynb` — English texts

5. Launch the app:
   ```bash
   streamlit run theory_llm_multi.py
   ```

## Usage

1. Select one or more databases from the sidebar.
2. Optionally filter results by author or source.
3. Enter your query in the text box and submit.
4. Review the LLM-generated response alongside the retrieved source passages.

Refer to the in-app **Help & Prompting Guide** (sidebar) for tips on writing effective queries.

## Contributors

- **Richard Freedman** — Haverford College
- **Daniel Russo-Batterham** — University of Melbourne
- **Charlie Cross** — Haverford College
- **Leo Ni** — Haverford College

## License

Copyright 2026 Richard Freedman, Daniel Russo-Batterham, Charlie Cross, Leo Ni

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

&nbsp;&nbsp;&nbsp;&nbsp;http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

---

*This project is part of ongoing research in computational musicology and digital humanities.*
