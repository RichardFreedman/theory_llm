import streamlit as st
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from typing_extensions import List, TypedDict
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from io import BytesIO
import pandas as pd
import os
from datetime import datetime

# Page config
st.set_page_config(page_title="Music Theory LLM RAG System", page_icon="🎵", layout="wide")

# Title (shown before login)
st.title("🎵 Historical Music Theory Query System")
st.markdown("### Multi-Database RAG with Author Filtering")

# API Key and App Password
try:
    openai_api_key = st.secrets["OPENAI_API_KEY"]
    app_password = st.secrets["APP_PASSWORD"]
except (KeyError, FileNotFoundError):
    openai_api_key = ""
    app_password = ""
    st.error("⚠️ OpenAI API key or APP_PASSWORD not found. Please add them to your Streamlit secrets.")

# Password authentication
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.subheader("🔒 Authentication Required")
    password_input = st.text_input("Enter password:", type="password", key="password_input")

    if st.button("Login"):
        if password_input == app_password:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("❌ Incorrect password. Please try again.")
    st.stop()

# Define State
class State(TypedDict):
    question: str
    context: List
    answer: str

st.write("This Streamlit application allows you to query a database of music theory texts using a large language model (LLM) with retrieval-augmented generation (RAG). Learn more about the system and how to write effective prompts with the tools at the left.") 

st.sidebar.header("📚 About the Project")

intro  = st.sidebar.checkbox("How to Use this Application", value=False, key="intro")

if intro:
    st.markdown("""
            * Enter the site password
            * Enter your question
            * select the number of text segments to retrieve
            * filter results by author
            * after a few minutes (depending on the number of segments and complexity of your question) you will see a response, followed by a list of the original source segments ('chunks'), and their authors and page references
            * use the button in the sidebar to download the results as a formatted PDF.
            """)
    
prompts = st.sidebar.checkbox("More about Writing AI Prompts", value=False, key="prompts")
if prompts:
    st.subheader("Tips for Writing Effective AI Prompts")   
    st.markdown("""
    In order to get the best results from AI language models, it's important to craft clear and specific prompts. Here are some tips:
    - **Be Specific**: Clearly state what you want the model to do. Vague prompts can lead to unpredictable results.  If you want the system to compare what different authors have to say on a topic, say so.
    - **Provide Context**: If your question relies on specific information, include that context in your prompt.  For instance you might provide a quotation from some other source that merits comment.
    - **Use Examples**: If applicable, provide examples of the type of response you're looking for.
    - **Specify Format of Output**: Perhaps you are asking for a list, some paragraphs, bullet points, etc.
    """)
    st.subheader("Some Prompt Examples")

    st.markdown("""
    * **Basic Question**: "What are the key elements of good music according to all the theorists in the database? Organize the results by author."
    * **Comparative Question**: "How do Thomas Morley and Elway Bevin differ in their views on counterpoint?" [Note for this you might want to filter results to just these two authors.  Use the dialogue in the sidebar at the left to do so.]
    * **Contextual Question**: "What does Thomas Morley say about the relationship between music and emotion?" [Again you might want to filter results to just this author.]
    * **Specific Format**: "Provide a bulleted list of the main points made by John Playford regarding dance music."
    """)

    st.subheader("How Many Text Segments ('Chunks') to Retrieve?")
    st.markdown("""
    The number of text segments (or 'chunks') you choose to retrieve can significantly impact the quality and relevance of the AI's response. There are some 1300 'segments' in the database.
    Here are some guidelines:
    - **Fewer Segments (1-10)**: This is useful for very specific questions where you expect a concise answer. The model will have less information to work with, which can lead to more focused responses but may miss broader context.
    - **Moderate Number of Segments (10-50)**: This range is often a good balance for general questions. It provides the model with enough context to generate a well-rounded answer without overwhelming it with too much information.
    - **Many Segments (50-100)**: This is suitable for complex questions that require comprehensive answers. However, be cautious as too much information can sometimes lead to confusion or less coherent responses.
    - **Consider the Question Type**: Tailor the number of segments based on whether your question is specific or broad.
    - **Experiment**: Don't hesitate to try different numbers of segments to see how it affects the quality of the responses.
    """)


rags  = st.sidebar.checkbox("More about RAG systems and LLMs", value=False, key="rags")
if rags:
    st.subheader("More about Retrieval-Augmented Generation (RAG) Systems")
    st.markdown("""
    * **Retrieval-Augmented Generation (RAG)** is a technique that combines the strengths of large language models (LLMs) with information retrieval systems. Instead of relying solely on the knowledge encoded in the
    parameters of the LLM, RAG systems first retrieve relevant documents from a database or corpus based on the user query. The retrieved documents are then used as context to generate more accurate and informed responses. This approach helps mitigate issues like hallucination, where LLMs generate plausible-sounding but incorrect or nonsensical answers.
    * The source documents are first **divided into segments (called 'chunks') of about 2000 characters** (not words).  The segments overlap with each other by about 200 characters to ensure that important context is not lost between segments.
    * These **segments are in turn passed to a LLM "embedding" system** (we use 'text-embedding-3-large' from OpenAI), which creates numerical representations of every segment. These representations capture the semantic meaning of the text, allowing for efficient similarity searches.  But they are very large:  each embedding has 3072 dimensions, representing a vast amount of information about the meaning of the text.
    * These representations (along with the original text of the segment and additional metadata about author, title, and date) are  stored in **'vector database'** (in our case: Chroma).
    * When you ask a question, **the system "retrieves" the most relevant text segments from this database**.  It does this with something called 'cosine similarity', a mathematical measure of similarity between vectors. Depending on the number of matching source texts you have requested (in our system this is from 1 to 10), we now have a set of 'contexts' that align with the ideas mentioned in your original query.
    * Now prepared with the question and relevant segments, **the system now "generates" an answer** based on those segments alone.  The prompt we use instructs the LLM to use only the information in the segments to answer the question, and not to 'hallucinate' information that is not present in the source texts.  The answer is generated with OpenAI's 'gpt-5-mini' model.  We could use a larger model, but this one is faster and less expensive, and seems to do a good job when provided with relevant context.
    * By combining retrieval with generation, RAG systems can provide more accurate, contextually relevant, and trustworthy answers to user queries.          
""")
    

credits = st.sidebar.checkbox("Credits", value=False, key="credits")
if credits:
    st.subheader("Developed by")
    st.markdown(""" 
    * Richard Freedman (Haverford College) 
    * Daniel Russo-Batterham (Melbourne University) 
    * Charlie Cross (Haverford College) 
    * Leo Ni (Haverford College))
    * Code at [GitHub Repository](https://github.com/RichardFreedman/theory_llm)""") 

st.sidebar.markdown("---")

# Sidebar - Database Selection at top
with st.sidebar:
    st.header("📚 Database Selection")
    use_english = st.checkbox("📖 English (TME)", value=True)
    use_italian = st.checkbox("📖 Italian (TMI)", value=True)
    use_latin = st.checkbox("📖 Latin (TML)", value=True)
    

# available sources
st.sidebar.checkbox("Show Available Sources", value=False, key="show_sources")
show_sources = st.session_state.get("show_sources", False)
if show_sources:
    if use_english:
        english_sources = pd.read_csv("english_html_metadata.csv")
        english_sources = english_sources[['author','title','date', 'citation']]
        st.subheader("TME Sources")
        st.dataframe(english_sources, width='stretch')

    if use_italian:
        italian_sources = pd.read_csv("italian_tmi_metadata.csv")
        italian_sources = italian_sources[['author','title','date', 'citation']]
        st.subheader("TMI Sources")
        st.dataframe(italian_sources, width='stretch')

    if use_latin:
        latin_sources = pd.read_csv("latin_tei_metadata.csv")
        latin_sources = latin_sources[['author','title','date', 'citation']]    
        st.subheader("TML Sources")
        st.dataframe(latin_sources, width='stretch')

st.sidebar.markdown("---")
# Database configurations
db_configs = []
if use_english:
    db_configs.append({
        "name": "English",
        "path": "../chroma_files/chroma-db_tme_english",
        "collection_name": "tme_english",
        "description": "TME "
    })
if use_italian:
    db_configs.append({
        "name": "Italian",
        "path": "../chroma_files/chroma-db_italian",
        "collection_name": "tmi_italian",
        "description": "Thesaurus Musicarum Italicarum"
    })
if use_latin:
    db_configs.append({
        "name": "Latin",
        "path": "../chroma_files/chroma-db_latin",
        "collection_name": "tml_latin",
        "description": "Thesaurus Musicarum Latinarum"
    })

if not db_configs:
    st.error("⚠️ Please select at least one database")
    st.stop()

# Track current DB selection to detect changes
current_db_selection = tuple(sorted([db['name'] for db in db_configs]))
if 'previous_db_selection' not in st.session_state:
    st.session_state.previous_db_selection = current_db_selection

# Load Vector Stores
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

@st.cache_resource
def load_single_vectorstore(db_path, collection_name, _embeddings):
    """Load a Chroma vector store with caching
    
    Note: _embeddings is prefixed with underscore to exclude from cache key
    """
    return Chroma(
        persist_directory=db_path,
        collection_name=collection_name,
        embedding_function=_embeddings
    )

# Load all selected databases
vector_stores = {}
for config in db_configs:
    with st.spinner(f"Loading {config['name']} database..."):
        vector_stores[config['name']] = load_single_vectorstore(
            config['path'],
            config['collection_name'],
            embeddings  # Now passing embeddings as parameter
        )

st.sidebar.success(f"✅ Loaded {len(vector_stores)} database(s)")

# # Load all selected databases
# vector_stores = {}
# for config in db_configs:
#     with st.spinner(f"Loading {config['name']} database..."):
#         st.write(f"**Attempting to load {config['name']}**")
        
#         # Check if path exists
#         import os
#         if os.path.exists(config['path']):
#             st.write(f"✅ Path exists: {config['path']}")
#             # List contents
#             contents = os.listdir(config['path'])
#             st.write(f"   Contents: {contents}")
#         else:
#             st.error(f"❌ Path does NOT exist: {config['path']}")
#             st.write(f"   Current working directory: {os.getcwd()}")
#             continue
        
#         try:
#             vector_stores[config['name']] = load_single_vectorstore(
#                 config['path'],
#                 config['collection_name'],
#                 embeddings
#             )
            
#             # Test the loaded database
#             st.write(f"🔍 Testing {config['name']} database...")
#             test_get = vector_stores[config['name']].get(limit=1)
            
#             st.write(f"   - IDs found: {len(test_get.get('ids', []))}")
#             st.write(f"   - Has documents: {len(test_get.get('documents', [])) > 0}")
#             st.write(f"   - Has metadata: {len(test_get.get('metadatas', [])) > 0}")
            
#             if test_get.get('metadatas') and len(test_get['metadatas']) > 0:
#                 st.write(f"   - Sample metadata keys: {list(test_get['metadatas'][0].keys())}")
            
#         except Exception as e:
#             st.error(f"❌ Error loading {config['name']}: {str(e)}")
#             import traceback
#             st.code(traceback.format_exc())

# st.success(f"✅ Loaded {len(vector_stores)} database(s)")

# Function to get date range from ALL selected databases
def get_date_range(vector_stores_dict):
    """Retrieve min and max dates from selected Chroma databases"""
    min_date = float('inf')
    max_date = float('-inf')
    for db_name, vector_store in vector_stores_dict.items():
        all_docs = vector_store.get()
        if 'metadatas' in all_docs:
            for metadata in all_docs['metadatas']:
                if metadata:
                    date_start = metadata.get('date_start')
                    date_end = metadata.get('date_end')
                    if date_start is not None:
                        try:
                            min_date = min(min_date, int(date_start))
                        except (ValueError, TypeError):
                            pass
                    if date_end is not None:
                        try:
                            max_date = max(max_date, int(date_end))
                        except (ValueError, TypeError):
                            pass
    # Fallback if no dates found
    if min_date == float('inf'):
        min_date = 500
    if max_date == float('-inf'):
        max_date = 1700
    # Round min down and max up to nearest century
    min_date = (int(min_date) // 100) * 100
    max_date = ((int(max_date) // 100) + 1) * 100
    return min_date, max_date

# Function to get unique authors from ALL selected databases, filtered by date range
def get_unique_authors(vector_stores_dict, date_range=None):
    """Retrieve all unique authors from selected Chroma databases, optionally filtered by date range"""
    authors = set()
    for db_name, vector_store in vector_stores_dict.items():
        all_docs = vector_store.get()
        if 'metadatas' in all_docs:
            for metadata in all_docs['metadatas']:
                if metadata and 'author' in metadata:
                    # If date_range is specified, check if document falls within range
                    if date_range is not None:
                        date_start = metadata.get('date_start')
                        date_end = metadata.get('date_end')
                        try:
                            doc_start = int(date_start) if date_start else 0
                            doc_end = int(date_end) if date_end else 9999
                        except (ValueError, TypeError):
                            doc_start, doc_end = 0, 9999
                        # Check if document's date range overlaps with selected range
                        if doc_end < date_range[0] or doc_start > date_range[1]:
                            continue  # Skip this document
                    authors.add(metadata['author'])
    return sorted(list(authors))

# Get date range from databases
db_min_date, db_max_date = get_date_range(vector_stores)

# Initialize date range in session state
if 'selected_date_range' not in st.session_state:
    st.session_state.selected_date_range = (db_min_date, db_max_date)

# Continue Sidebar - Date Filter
with st.sidebar:
    st.header("📅 Filter by Date")
    st.write(f"Sources span {db_min_date} - {db_max_date}")

    selected_date_range = st.slider(
        "Select date range:",
        min_value=db_min_date,
        max_value=db_max_date,
        value=st.session_state.selected_date_range,
        step=100,
        help="Filter sources by publication date range"
    )

    # Update session state
    if selected_date_range != st.session_state.selected_date_range:
        st.session_state.selected_date_range = selected_date_range
        # Reset authors when date changes
        if 'selected_authors' in st.session_state:
            del st.session_state.selected_authors

    st.markdown("---")

# Build the author list (filtered by date range)
available_authors = get_unique_authors(vector_stores, date_range=selected_date_range)

# Reset selected authors if database selection changed
if current_db_selection != st.session_state.previous_db_selection:
    st.session_state.selected_authors = available_authors.copy()
    st.session_state.previous_db_selection = current_db_selection

if 'selected_authors' not in st.session_state:
    st.session_state.selected_authors = available_authors.copy()

# Ensure selected authors are still valid after date filter change
st.session_state.selected_authors = [
    a for a in st.session_state.selected_authors if a in available_authors
]
if not st.session_state.selected_authors:
    st.session_state.selected_authors = available_authors.copy()

# Continue Sidebar - Author Selection and Settings
with st.sidebar:
    st.header("✏️ Filter by Author")
    st.write(f"Authors in selected date range ({len(available_authors)} available):")

    if st.button("Select All Authors"):
        st.session_state.selected_authors = available_authors.copy()
        st.rerun()

    selected_authors = st.multiselect(
        "Choose authors:",
        options=available_authors,
        default=st.session_state.selected_authors,
        help="Select which authors to include in search results"
    )

    # Update session state when selection changes
    if selected_authors != st.session_state.selected_authors:
        st.session_state.selected_authors = selected_authors

    st.markdown("---")

    # Retrieval settings
    st.header("⚙️ Retrieval Settings")
    k = st.slider(
        "Number of segments per database:",
        min_value=1,
        max_value=20,
        value=5,
        help="More segments = more context but slower response"
    )

    st.markdown("---")
    st.subheader("📊 Database Info")
    st.write(f"**Active Databases:** {len(vector_stores)}")
    st.write(f"**Date Range:** {selected_date_range[0]} - {selected_date_range[1]}")
    st.write(f"**Authors in Range:** {len(available_authors)}")
    st.write(f"**Selected Authors:** {len(selected_authors)}")

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Prompt template
system_prompt = """You are an expert in historical music theory and musicology.

You are also familiar with medieval Latin, and various early modern forms of English, Italian and French.

Use the following context passages to answer the question.

IMPORTANT: Each text passage is labeled with a Source number (e.g., "Source 1", "Source 2"), author, title, and date.
When citing passages, always reference them by their Source number (e.g., "Source 1", "Source 5") so readers can
find the exact passage. Also mention the author's name when making claims about their ideas.

Include short quotations from the passages to support your statements, with key words from the original text
and translation when appropriate.

If you don't know the answer based on the provided context, say that you don't know.
Use three sentences maximum and keep the answer concise but informative.

Context:
{context}

Question: {question}

Provide a detailed answer with references to specific Source numbers and authors."""

prompt = ChatPromptTemplate.from_template(system_prompt)

# Retrieval function
def retrieve(state: State):
    """
    Unless otherwise instructed to restrict things to just the English, Italian, or Latin vector stores, retrieve documents from ALL selected vector stores and filter by selected authors and date range.
    """
    question = state["question"]
    selected_authors_list = st.session_state.get('selected_authors', available_authors)
    date_range = st.session_state.get('selected_date_range', (db_min_date, db_max_date))

    # Display filter info
    filter_info = []
    if date_range != (db_min_date, db_max_date):
        filter_info.append(f"dates {date_range[0]}-{date_range[1]}")
    if selected_authors_list and len(selected_authors_list) < len(available_authors):
        filter_info.append(f"{len(selected_authors_list)} author(s)")

    if filter_info:
        st.write(f"**Filtering by:** {', '.join(filter_info)}")
    else:
        st.write("*Searching all sources*")

    # Retrieve from ALL selected databases
    all_docs = []
    for db_name, vector_store in vector_stores.items():
        retriever = vector_store.as_retriever(search_kwargs={"k": k})
        docs = retriever.invoke(question)
        all_docs.extend(docs)

    st.write(f"Retrieved {len(all_docs)} total segments from {len(vector_stores)} database(s)")

    # Filter by date range
    def doc_in_date_range(doc, date_range):
        try:
            doc_start = int(doc.metadata.get('date_start', 0))
            doc_end = int(doc.metadata.get('date_end', 9999))
        except (ValueError, TypeError):
            return True  # Include if date parsing fails
        # Check if document's date range overlaps with selected range
        return not (doc_end < date_range[0] or doc_start > date_range[1])

    RAG_retrieved_docs = [doc for doc in all_docs if doc_in_date_range(doc, date_range)]

    if len(RAG_retrieved_docs) < len(all_docs):
        st.write(f"After date filtering: {len(RAG_retrieved_docs)} segments")

    # Filter by selected authors if specified
    if selected_authors_list and len(selected_authors_list) < len(available_authors):
        RAG_retrieved_docs = [
            doc for doc in RAG_retrieved_docs
            if doc.metadata.get('author') in selected_authors_list
        ]
        st.write(f"After author filtering: {len(RAG_retrieved_docs)} segments from selected authors")

    return {"context": RAG_retrieved_docs}

# Generation function with author grouping
def generate_with_author_grouping(state: State):
    """
    Generate response with chunks grouped by author for better comparison.
    Sources are numbered globally (Source 1, Source 2, etc.) to match the UI display.
    """
    # First, assign global source numbers to each document (matching UI order)
    docs_with_numbers = list(enumerate(state["context"], 1))

    # Group documents by author while preserving global source numbers
    author_groups = {}
    for source_num, doc in docs_with_numbers:
        author = doc.metadata.get('author', 'Unknown Author')
        if author not in author_groups:
            author_groups[author] = []
        author_groups[author].append((source_num, doc))

    # Format context with author groupings and global source numbers
    context_parts = []
    for author, numbered_docs in author_groups.items():
        author_section = f"\n=== {author} ===\n"
        for source_num, doc in numbered_docs:
            title = doc.metadata.get('title', 'Unknown Title')
            date = doc.metadata.get('date', 'Unknown')
            page_range = doc.metadata.get('page_range', 'Unknown')
            author_section += f"\n[Source {source_num}] '{title}' ({date}), pp. {page_range}:\n{doc.page_content}\n"
        context_parts.append(author_section)

    # Join all author sections
    formatted_context = "\n".join(context_parts)

    # Generate response
    messages = prompt.invoke({"context": formatted_context, "question": state["question"]})
    response = llm.invoke(messages)

    return {"answer": response.content}

# Build LangGraph
graph_builder = StateGraph(State).add_sequence([retrieve, generate_with_author_grouping])
graph_builder.add_edge(START, "retrieve")
graph_builder.add_edge("generate_with_author_grouping", END)
graph = graph_builder.compile()

# PDF Generation
def create_pdf(question, answer, context_docs, selected_dbs, selected_authors_list, date_range=None):
    buffer = BytesIO()
    pdf_doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=72, leftMargin=72,
                                topMargin=72, bottomMargin=18)

    elements = []
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'],
                                 fontSize=24, textColor='darkblue', spaceAfter=30)

    heading_style = ParagraphStyle('CustomHeading', parent=styles['Heading2'],
                                   fontSize=14, textColor='darkblue', spaceAfter=12, spaceBefore=12)

    body_style = ParagraphStyle('CustomBody', parent=styles['BodyText'],
                               fontSize=11, alignment=TA_JUSTIFY, spaceAfter=12)

    # Title
    elements.append(Paragraph("Historical Music Theory Query Report", title_style))
    elements.append(Spacer(1, 0.2*inch))

    # Metadata
    date_range_str = f"{date_range[0]} - {date_range[1]}" if date_range else "All Dates"
    metadata_text = f"""
    <b>Date:</b> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}<br/>
    <b>Databases:</b> {', '.join(selected_dbs)}<br/>
    <b>Date Range:</b> {date_range_str}<br/>
    <b>Authors:</b> {', '.join(selected_authors_list) if len(selected_authors_list) < len(available_authors) else 'All Authors'}<br/>
    <b>Segments:</b> {len(context_docs)}
    """
    elements.append(Paragraph(metadata_text, body_style))
    elements.append(Spacer(1, 0.3*inch))

    # Query
    elements.append(Paragraph("Query", heading_style))
    elements.append(Paragraph(question, body_style))
    elements.append(Spacer(1, 0.2*inch))

    # Answer
    elements.append(Paragraph("Answer", heading_style))
    for para in answer.split('\n\n'):
        if para.strip():
            elements.append(Paragraph(para, body_style))

    elements.append(PageBreak())

    # Source Documents
    elements.append(Paragraph("Source Documents", heading_style))

    for i, source_doc in enumerate(context_docs, 1):
        elements.append(Paragraph(f"<b>Source {i}</b>", heading_style))

        metadata = source_doc.metadata
        meta_text = f"<b>Author:</b> {metadata.get('author', 'Unknown')}<br/>"
        meta_text += f"<b>Title:</b> {metadata.get('title', 'Unknown')}<br/>"
        meta_text += f"<b>Date:</b> {metadata.get('date', 'Unknown')}<br/>"
        meta_text += f"<b>Page:</b> {metadata.get('page_range', 'Unknown')}<br/>"
        meta_text += f"<b>Source:</b> {metadata.get('citation', 'Unknown')}<br/>"

        elements.append(Paragraph(meta_text, body_style))
        elements.append(Spacer(1, 0.1*inch))

        content = source_doc.page_content.replace('\n', '<br/>')
        elements.append(Paragraph(content, body_style))
        elements.append(Spacer(1, 0.2*inch))

    pdf_doc.build(elements)
    buffer.seek(0)
    return buffer

# Initialize session state for results
if 'last_result' not in st.session_state:
    st.session_state.last_result = None
if 'last_query' not in st.session_state:
    st.session_state.last_query = None
if 'last_db_names' not in st.session_state:
    st.session_state.last_db_names = None

# Query Interface
st.markdown("---")
st.subheader("💬 Ask Your Question")

user_query = st.text_area(
    "Enter your question:",
    height=100,
    placeholder="e.g., What do different theorists say about the origins of music?"
)

submit = st.button("🔍 Search", type="primary")

if submit:
    if user_query:
        with st.spinner("🔄 Processing query..."):
            result = graph.invoke({"question": user_query})
            # Store results in session state
            st.session_state.last_result = result
            st.session_state.last_query = user_query
            st.session_state.last_db_names = [db['name'] for db in db_configs]
    else:
        st.warning("⚠️ Please enter a question")

# Display results if available
if st.session_state.last_result:
    result = st.session_state.last_result

    st.markdown("### 📝 Answer")
    st.write(result["answer"])

    st.markdown("---")
    st.markdown("### 📚 Source Documents")

    for i, doc in enumerate(result["context"], 1):
        metadata = doc.metadata
        author = metadata.get('author', 'Unknown')
        title = metadata.get('title', 'Unknown')
        date_raw = metadata.get('date', 'Unknown')
        date_str = str(date_raw)
        if len(date_str) == 4:
            date = date_raw
        elif 'th' in date_str:
            date = date_str + ' century'
        else:
            date = date_raw
        citation = metadata.get('citation', 'Unknown Source')
        pages = metadata.get('page_range', 'Unknown Page')
        with st.expander(f"📄 Source {i}: {author} - {title}"):
            # Formatted metadata block
            st.markdown(f"""
**Author:** {author}<br/>
**Title:** {title}<br/>
**Date:** {date}<br/>
**Page(s):** {pages}<br/>
**Citation:** {citation}
""", unsafe_allow_html=True)
            st.markdown("---")
            st.markdown("**Passage:**")
            st.write(doc.page_content)

# PDF Download in Sidebar (always available after a query)
with st.sidebar:
    st.markdown("---")
    st.header("📄 Export Results")
    if st.session_state.last_result:
        pdf = create_pdf(
            st.session_state.last_query,
            st.session_state.last_result["answer"],
            st.session_state.last_result["context"],
            st.session_state.last_db_names or [db['name'] for db in db_configs],
            selected_authors,
            date_range=selected_date_range
        )
        st.download_button(
            "📥 Download PDF Report",
            pdf,
            file_name=f"query_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf",
            type="primary"
        )
    else:
        st.write("*Run a query to enable PDF export*")