import streamlit as st
import os
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from typing import List, TypedDict
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import START, StateGraph
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle, ListStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, ListFlowable, ListItem
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from io import BytesIO
from datetime import datetime
import re
import pandas as pd


st.set_page_config(page_title='Ask the Italian Music Theorists', page_icon='🔎')

st.sidebar.header('About this App 🔎')
st.title('🔎 Ask the Music Theorists')
st.write("This Streamlit application allows you to query a database of music theory texts using a large language model (LLM) with retrieval-augmented generation (RAG). Learn more about the system and how to write effective prompts with the tools at the left.") 
         
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

sources  = st.sidebar.checkbox("Our Music Theory Treatises", value=False, key="sources")
if sources:
    st.subheader("Our Music Theory Treatises")
    st.markdown("""
    This project uses the following music theory treatises as its source material.  All of them are drawn from the [Early English Books Online Text Creation Partnership](https://textcreationpartnership.org/tcp-texts/eebo-tcp-early-english-books-online/), and used by permission.  The TCP editions are full-text html transcriptions of the original sources.  Our code extracts textual data (but not the musical examples) from these transcriptions for processing with LLM tools.""")
    # Create DataFrame from the provided data
    data = {
        'Author': [
            'Morley, Thomas, 1557-1603?',
            'Robinson, Thomas, fl. 1589-1609',
            'Ravenscroft, Thomas, 1592?-1635?',
            'Bevin, Elway, ca. 1554-1638',
            'Descartes, René, 1596-1650',
            'Playford, John, 1623-1686?',
            'Le Roy, Adrian, ca. 1520-1598',
            'Bathe, William, 1564-1614',
            'Ornithoparchus, Andreas, 16th cent.'
        ],
        'Title': [
            'A plaine and easie introduction to practicall musicke',
            'New citharen lessons with perfect tunings of the same',
            'A briefe discourse of the true (but neglected) vse of charact\'ring the degrees',
            'A briefe and short instruction of the art of musicke',
            'Renatus Des-Cartes excellent compendium of musick',
            'A breefe introduction to the skill of musick for song & violl',
            'A briefe and plaine instruction to set all musicke of eight diuers tunes',
            'A briefe introduction to the skill of song',
            'Andreas Ornithoparcus his Micrologus'
        ],
        'Date': ['1596','1609', '1614', '1623', '1653', '1654', '1574', '1596', '1609'],
        'URL': [
            'https://quod.lib.umich.edu/e/eebo/A07753.0001.001?rgn=main;view=fulltext',
            'https://quod.lib.umich.edu/e/eebo/A10856.0001.001?rgn=main;view=fulltext',
            'https://quod.lib.umich.edu/e/eebo/A10477.0001.001?rgn=main;view=fulltext',
            'https://quod.lib.umich.edu/e/eebo/A09578.0001.001?rgn=main;view=fulltext',
            'https://quod.lib.umich.edu/e/eebo/A35748.0001.001?rgn=main;view=fulltext',
            'https://quod.lib.umich.edu/e/eebo/A55042.0001.001?rgn=main;view=fulltext',
            'https://quod.lib.umich.edu/e/eebo/A05334.0001.001?rgn=main;view=fulltext',
            'https://quod.lib.umich.edu/e/eebo/A05729.0001.001?rgn=main;view=fulltext',
            'https://quod.lib.umich.edu/e/eebo/A08534.0001.001?rgn=main;view=fulltext'
        ]
    }

    df = pd.DataFrame(data)

    # Display the table with full width
    # st.dataframe(df, use_container_width=True)
    # Method 1: Create a modified DataFrame with clickable links using HTML
    def create_clickable_dataframe(df):
        """Convert URLs in DataFrame to clickable HTML links"""
        df_display = df.copy()
        
        # Create clickable links for the URL column
        df_display['URL'] = df_display['URL'].apply(
            lambda x: f'<a href="{x}" target="_blank">View Source</a>'
        )
        
        return df_display

    # Display the table with clickable links using HTML
    df_clickable = create_clickable_dataframe(df)

    # Convert DataFrame to HTML and display with clickable links
    html_table = df_clickable.to_html(escape=False, index=False)
    st.markdown(html_table, unsafe_allow_html=True)

    st.markdown("---")

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

# Get API key from Streamlit secrets


# try:
#     # openai_api_key = st.secrets[OPENAI_API_KEY]
#     # app_password = st.secrets["APP_PASSWORD"]
# except (KeyError, FileNotFoundError):
#     openai_api_key = ""
#     # app_password = ""
#     st.error("⚠️ OpenAI API key or APP_PASSWORD not found. Please add them to your Streamlit secrets.")

# Password authentication
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

# if not st.session_state.authenticated:
#     st.subheader("🔒 Authentication Required")
#     password_input = st.text_input("Enter password:", type="password", key="password_input")
    
#     if st.button("Login"):
#         if password_input == app_password:
#             st.session_state.authenticated = True
#             st.rerun()
#         else:
#             st.error("❌ Incorrect password. Please try again.")
#     st.stop()



st.sidebar.header('Select Language Idiom ⚙️')
st.sidebar.write("Choose the language style for the LLM's responses. By default, the system will respond to you in modern English (or French, German, etc.), depending on the language of your query.  Alternatively, you can require the system to respond in a style approximating Elizabethan English, which resembles the tone of our original sources.  ")
language = st.sidebar.selectbox("Select Language", options=["Modern Language", "Period English"], index=0, disabled=False)

# Function to get unique authors
def get_unique_authors(vector_store):
    """Retrieve all unique authors from the Chroma database"""
    all_docs = vector_store.get()
    authors = set()
    if 'metadatas' in all_docs:
        for metadata in all_docs['metadatas']:
            if metadata and 'author' in metadata:
                authors.add(metadata['author'])
    return sorted(list(authors))


def create_pdf(question, answer, context_docs, language):
    """Generate a formatted PDF with query results"""
    
    def format_text_for_pdf(text):
        """Convert markdown-style formatting to HTML for ReportLab"""
        # Escape special XML characters first
        text = text.replace('&', '&amp;')
        text = text.replace('<', '&lt;')
        text = text.replace('>', '&gt;')
        
        # Replace all types of dashes and hyphens with regular hyphen
        text = text.replace('—', '-')  # em dash
        text = text.replace('–', '-')  # en dash
        text = text.replace('‑', '-')  # non-breaking hyphen
        text = text.replace('‐', '-')  # hyphen (Unicode)
        text = text.replace('−', '-')  # minus sign
        
        # Replace smart quotes with regular quotes
        text = text.replace('"', '"')  # left double quote
        text = text.replace('"', '"')  # right double quote
        text = text.replace(''', "'")  # left single quote
        text = text.replace(''', "'")  # right single quote
        
        # Convert **bold** to <b>bold</b>
        text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
        
        # Convert *italic* to <i>italic</i>
        text = re.sub(r'\*(.+?)\*', r'<i>\1</i>', text)
        
        return text
    
    def parse_text_to_flowables(text, body_style):
        """Parse text and return list of flowables including proper bullet lists"""
        lines = text.split('\n')
        flowables = []
        current_list_items = []
        current_paragraph = []
        
        for line in lines:
            stripped = line.strip()
            
            # Check if line starts with a bullet point
            if stripped.startswith(('- ', '* ', '• ')):
                # If we have a paragraph in progress, add it first
                if current_paragraph:
                    para_text = '<br/>'.join(current_paragraph)
                    flowables.append(Paragraph(format_text_for_pdf(para_text), body_style))
                    current_paragraph = []
                
                # Add to current list
                content = stripped[2:].strip()
                current_list_items.append(ListItem(Paragraph(format_text_for_pdf(content), body_style), leftIndent=20, spaceBefore=6, spaceAfter=6))
            else:
                # If we have list items in progress, create the list
                if current_list_items:
                    flowables.append(ListFlowable(current_list_items, bulletType='bullet', start='•'))
                    current_list_items = []
                
                # Add to current paragraph
                if stripped:
                    current_paragraph.append(stripped)
                elif current_paragraph:
                    # Empty line - end current paragraph
                    para_text = '<br/>'.join(current_paragraph)
                    flowables.append(Paragraph(format_text_for_pdf(para_text), body_style))
                    flowables.append(Spacer(1, 0.1*inch))
                    current_paragraph = []
        
        # Add any remaining paragraph
        if current_paragraph:
            para_text = '<br/>'.join(current_paragraph)
            flowables.append(Paragraph(format_text_for_pdf(para_text), body_style))
        
        # Add any remaining list
        if current_list_items:
            flowables.append(ListFlowable(current_list_items, bulletType='bullet', start='•'))
        
        return flowables
    
    buffer = BytesIO()
    pdf_doc = SimpleDocTemplate(buffer, pagesize=letter,
                           rightMargin=72, leftMargin=72,
                           topMargin=72, bottomMargin=18)
    
    # Container for the 'Flowable' objects
    elements = []
    
    # Define styles
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        textColor='#1f4788',
        spaceAfter=30,
        alignment=TA_CENTER
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor='#1f4788',
        spaceAfter=12,
        spaceBefore=12
    )
    
    subheading_style = ParagraphStyle(
        'CustomSubheading',
        parent=styles['Heading3'],
        fontSize=12,
        textColor='#2c5aa0',
        spaceAfter=8,
        spaceBefore=8
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['BodyText'],
        fontSize=11,
        spaceAfter=12,
        alignment=TA_LEFT,
        bulletIndent=20,
        leftIndent=20,
        spaceBefore=6
    )
    
    # Title
    elements.append(Paragraph("Music Theory Query Results", title_style))
    elements.append(Spacer(1, 0.2*inch))
    
    # Metadata
    date_str = datetime.now().strftime("%B %d, %Y at %I:%M %p")
    elements.append(Paragraph(f"<b>Generated:</b> {date_str}", body_style))
    elements.append(Paragraph(f"<b>Language Style:</b> {language}", body_style))
    elements.append(Spacer(1, 0.3*inch))
    
    # Question
    elements.append(Paragraph("Question", heading_style))
    question_flowables = parse_text_to_flowables(question, body_style)
    elements.extend(question_flowables)
    elements.append(Spacer(1, 0.2*inch))
    
    # Answer
    elements.append(Paragraph("Answer", heading_style))
    answer_flowables = parse_text_to_flowables(answer, body_style)
    elements.extend(answer_flowables)
    elements.append(Spacer(1, 0.3*inch))
    
    # Source Documents
    elements.append(Paragraph("Source Documents", heading_style))
    elements.append(Spacer(1, 0.1*inch))
    
    for idx, doc in enumerate(context_docs, 1):
        metadata = doc.metadata
        
        elements.append(Paragraph(f"Source {idx}", subheading_style))
        elements.append(Paragraph(f"<b>Author:</b> {metadata.get('author', 'Unknown')}", body_style))
        # elements.append(Paragraph(f"<b>File Name:</b> {metadata.get('source_file', 'Unknown')}", body_style))
        elements.append(Paragraph(f"<b>Title:</b> {metadata.get('title', 'Unknown')}", body_style))
        elements.append(Paragraph(f"<b>Source:</b> {metadata.get('citation', 'Unknown')}", body_style))
        elements.append(Paragraph(f"<b>Page Number:</b> {metadata.get('page_number', 1)}", body_style))
        elements.append(Paragraph(f"<b>Original Source Passage:</b>", body_style))
        passage_flowables = parse_text_to_flowables(doc.page_content, body_style)
        elements.extend(passage_flowables)
        elements.append(Spacer(1, 0.2*inch))
    
    # Build PDF
    pdf_doc.build(elements)
    buffer.seek(0)
    return buffer


if not openai_api_key.startswith("sk-"):
    st.warning("Please enter your OpenAI API key **in the sidebar**.", icon="👈")
else:
    os.environ["OPENAI_API_KEY"] = openai_api_key
    llm = init_chat_model('gpt-5-mini', model_provider='openai')

    embeddings = OpenAIEmbeddings(model='text-embedding-3-large')

    vector_store = Chroma(
        collection_name='HTML_samples_italian',
        embedding_function=embeddings,
        persist_directory=f'{Path.cwd()}/chroma-db_italian'
    )

    # Get available authors and create multiselect
    available_authors = get_unique_authors(vector_store)
    # Initialize session state if needed
    if 'selected_authors' not in st.session_state:
        st.session_state.selected_authors = available_authors.copy()
        
    st.sidebar.header('Filter Authors ✍️')
    st.sidebar.write("By default, all authors are included in the search; you can select specific authors to narrow the results.")

    # Add a simple "Select All" button
    if st.sidebar.button("Select All Authors"):
        st.session_state.selected_authors = available_authors.copy()
        st.rerun()

    # Your existing multiselect with key parameter added
    selected_authors = st.sidebar.multiselect(
        "Filter by Author",
        options=available_authors,
        default=st.session_state.selected_authors,
        key="selected_authors",  # This syncs with session state
        help="Select which authors to search"
    )

        # st.sidebar.write("By default, all authors are included in the search; you can select specific authors to narrow the results.")
        # selected_authors = st.sidebar.multiselect(
        #     "Filter by Author",
        #     options=available_authors,
        #     default=available_authors,  # All authors selected by default
        #     help="Select which authors to search"
        # )

    class State(TypedDict):
        question: str
        context: List[Document]
        answer: str
        k: int
    
    # if language == "Modern English":
    #     prompt = ChatPromptTemplate.from_messages([
    #         ("system", "You an expert in music theory.  All your answers should read in Modern English. Use only the information provided in the context below to answer the question. If the answer is not in the context, do not fabricate an answer.  Instead explain that the information is not available.'"),
    #         ("human", "Context:\n{context}\n\nQuestion: {question}")
    #     ])
    # else:
    #     prompt = ChatPromptTemplate.from_messages([
    #         ("system", "You an expert in music theory, and are also familiar with Elizabethan English.  All your answers should read in this style. Use only the information provided in the context below to answer the question. If the answer is not in the context, do not fabricate an answer.  Instead explain that the information is not available.'"),
    #         ("human", "Context:\n{context}\n\nQuestion: {question}")
    #     ])
    if language == "Modern Language":
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert in music theory of the sixteenth and seventeenth centuries. Your response should be in the same language as that of the human prompt.  For instance, if in English, then use that.  If in French, the respond in French. It will be helpful to briefly quote and cite the page numbers of the original sources in as needed in your response. 
            Use only the information provided in the context below to answer the question. 
            
            IMPORTANT: Each text passage is clearly labeled with its author. When relevant to the question, 
            compare and contrast what different authors say about the topic. Cite specific authors by name 
            when referencing their ideas, and quote them briefly to support your response. If the answer is not in the context, do not fabricate an answer. 
            Instead explain that the information is not available.
             
            """),
            ("human", "Context:\n{context}\n\nQuestion: {question}")
        ])
    else:
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert in music theory of the sixteenth and seventeenth centuries, and are also familiar with Elizabethan English. 
            All your answers should read in this style. Use only the information provided in the context below 
            to answer the question.
            
            IMPORTANT: Each text passage is clearly labeled with its author. When relevant to the question, 
            compare and contrast what different authors say about the topic. Cite specific authors by name 
            when referencing their ideas. If the answer is not in the context, do not fabricate an answer. 
            Instead explain that the information is not available."""),
            ("human", "Context:\n{context}\n\nQuestion: {question}")
        ])

    # 
    def retrieve(state: State):
        """
        Retrieve documents from vector store and filter by selected authors.
        Displays selected authors in the main Streamlit window.
        """
        # Get selected authors from Streamlit session state
        selected_authors = st.session_state.get('selected_authors', available_authors)
        
        # Display the selected authors in the main window
        if selected_authors and len(selected_authors) < len(available_authors):
            st.subheader("Selected Authors:")
            # Display as a bulleted list
            for author in selected_authors:
                st.write(f"• {author}")
        else:
            st.write("*Showing results from all authors")
        
        # Get initial results from vector store
        RAG_retrieved_docs = vector_store.similarity_search(state['question'], k=state['k'])
        
        # Filter by selected authors if any are specified
        if selected_authors and len(selected_authors) < len(available_authors):
            filtered_docs = [
                doc for doc in RAG_retrieved_docs 
                if doc.metadata.get('author') in selected_authors
            ]
            # Display count of filtered results
            st.write(f"Found {len(filtered_docs)} segments from selected authors out of {len(RAG_retrieved_docs)} total documents")
            return {"context": filtered_docs}
        else:
            # Display count of all results
            st.write(f"Found {len(RAG_retrieved_docs)} segments from all authors")
            return {"context": RAG_retrieved_docs}

    # def generate(state: State):
    #     docs_str = '\n\n'.join([doc.page_content for doc in state['context']])
    #     message = prompt.invoke({"question": state["question"], "context": docs_str})
    #     response = llm.invoke(message)
    #     return {'answer': response.content}



    # graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    # graph_builder.add_edge(START, 'retrieve')
    # graph = graph_builder.compile()

    def generate_with_author_grouping(state: State):
        """
        Generate response with chunks grouped by author for better comparison.
        """
        # Group documents by author
        author_groups = {}
        for doc in state['context']:
            author = doc.metadata.get('author', 'Unknown Author')
            if author not in author_groups:
                author_groups[author] = []
            author_groups[author].append(doc)
        
        # Format context with author groupings
        context_parts = []
        for author, docs in author_groups.items():
            author_section = f"\n=== {author} ===\n"
            
            for i, doc in enumerate(docs, 1):
                title = doc.metadata.get('title', 'Unknown Title')
                author_section += f"\nPassage {i} from '{title}':\n{doc.page_content}\n"
            
            context_parts.append(author_section)
        
        # Join all author sections
        docs_str = '\n'.join(context_parts)
        
        # Invoke the prompt with the grouped context
        message = prompt.invoke({"question": state["question"], "context": docs_str})
        response = llm.invoke(message)
        return {'answer': response.content}

    # Correct graph builder using the grouped approach
    graph_builder = StateGraph(State).add_sequence([retrieve, generate_with_author_grouping])
    graph_builder.add_edge(START, 'retrieve')
    graph = graph_builder.compile()

    # Initialize session state
    if 'response' not in st.session_state:
        st.session_state.response = None
    if 'last_question' not in st.session_state:
        st.session_state.last_question = None
    if 'last_language' not in st.session_state:
        st.session_state.last_language = None

    with st.form("my_form"):
        question = st.text_area(
            "Enter question:",
            "What are the key elements of good music?",
        )
        k = st.text_area(
            "Enter number of text segments (1-100) to fetch from the database:",
            "10",
        )
        submitted = st.form_submit_button("Submit Query")

    if submitted:
        response = graph.invoke({
            "question": question,
            "k": int(k)
        })
        # Store in session state
        st.session_state.response = response
        st.session_state.last_question = question
        st.session_state.last_language = language
    
    # Display results if they exist
    if st.session_state.response:
        st.info(st.session_state.response['answer'])
        
        st.info('The following are the text chunks fetched from your sources:')
        for doc in st.session_state.response['context']:
            metadata = doc.metadata
            st.info(f"Author: {metadata.get('author', 'Unknown')}")
            # st.info(f"File Name: {metadata.get('source_file', 'Unknown')}")
            st.info(f"Title: {metadata.get('title', 'Unknown')}")
            st.info(f"Source: {metadata.get('citation', 'Unknown')}")
            st.info(f"Page Number: {metadata.get('page_number', 1)}")
            st.info(f"Original Source Passage: {doc.page_content}")
            st.info("---")
        
        # Generate PDF
        pdf_buffer = create_pdf(
            question=st.session_state.last_question,
            answer=st.session_state.response['answer'],
            context_docs=st.session_state.response['context'],
            language=st.session_state.last_language
        )
        
        # Create download button
        st.sidebar.download_button(
            label="📥 Download Results as PDF",
            data=pdf_buffer,
            file_name=f"music_theory_query_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf",
            type="primary"
        )