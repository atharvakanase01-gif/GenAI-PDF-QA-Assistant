import streamlit as st
from PyPDF2 import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama

# PAGE SETTINGS
st.set_page_config(
    page_title="GenAI PDF Q&A Assistant",
    page_icon="🤖",
    layout="wide"
)

# SESSION STATE FOR CHAT HISTORY
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# CUSTOM CSS
st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

.title {
    font-size: 50px;
    font-weight: bold;
    text-align: center;
    color: #4FC3F7;
    margin-bottom: 10px;
}

.subtitle {
    text-align: center;
    color: #BBBBBB;
    margin-bottom: 30px;
    font-size: 18px;
}

.card {
    background-color: #1E1E1E;
    padding: 25px;
    border-radius: 20px;
    margin-bottom: 20px;
    box-shadow: 0px 0px 12px rgba(255,255,255,0.08);
}

.chat-box {
    background-color: #262730;
    padding: 20px;
    border-radius: 15px;
    margin-top: 15px;
    border-left: 5px solid #4FC3F7;
}

.user-box {
    background-color: #1B4332;
    padding: 15px;
    border-radius: 12px;
    margin-top: 10px;
}

.ai-box {
    background-color: #262730;
    padding: 15px;
    border-radius: 12px;
    margin-top: 10px;
}

.footer {
    text-align: center;
    color: gray;
    margin-top: 40px;
}

</style>
""", unsafe_allow_html=True)

# TITLE
st.markdown(
    '<div class="title">🤖 GenAI PDF Q&A Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Transform your PDFs into intelligent AI conversations using Local LLMs and RAG.</div>',
    unsafe_allow_html=True
)

# SIDEBAR
with st.sidebar:

    st.header("⚙️ Settings")

    model_name = st.selectbox(
        "Choose AI Model",
        ["tinyllama", "phi3"]
    )

    st.markdown("---")

    st.subheader("🚀 Features")

    st.write("✅ PDF Upload")
    st.write("✅ AI Question Answering")
    st.write("✅ Local LLM")
    st.write("✅ Vector Search")
    st.write("✅ Chat History")
    st.write("✅ Modern UI")

    st.markdown("---")

    st.subheader("💡 Example Questions")

    st.write("• Summarize this document")
    st.write("• What skills are mentioned?")
    st.write("• Explain the projects section")
    st.write("• What experience is mentioned?")
    st.write("• What technologies are used?")

    st.markdown("---")

    st.subheader("🧠 Tech Stack")

    st.write("• Streamlit")
    st.write("• LangChain")
    st.write("• FAISS")
    st.write("• Ollama")

# FILE UPLOADER
uploaded_file = st.file_uploader(
    "📄 Upload PDF Document",
    type="pdf"
)

# MAIN LOGIC
if uploaded_file is not None:

    # READ PDF
    pdf_reader = PdfReader(uploaded_file)

    text = ""

    for page in pdf_reader.pages:

        extracted_text = page.extract_text()

        if extracted_text:
            text += extracted_text

    # TEXT SPLITTER
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = text_splitter.split_text(text)

    # EMBEDDINGS
    embeddings = OllamaEmbeddings(model=model_name)

    # VECTOR DATABASE
    vectorstore = FAISS.from_texts(chunks, embeddings)

    st.success("✅ PDF processed successfully!")

    # DASHBOARD
    col1, col2 = st.columns([2, 1])

    # LEFT SIDE
    with col1:

        st.markdown('<div class="card">', unsafe_allow_html=True)

        st.subheader("💬 Ask Questions")

        user_question = st.text_input(
            "Ask anything about your PDF"
        )

        if user_question:

            with st.spinner("🤖 AI is thinking..."):

                # SEARCH DOCUMENTS
                docs = vectorstore.similarity_search(user_question)

                context = ""

                for doc in docs:
                    context += doc.page_content

                # LOAD MODEL
                llm = Ollama(model=model_name)

                # PROMPT
                prompt = f"""
                You are a helpful AI assistant.

                Answer the question using only the PDF context below.

                Context:
                {context}

                Question:
                {user_question}
                """

                # RESPONSE
                response = llm.invoke(prompt)

                # SAVE CHAT HISTORY
                st.session_state.chat_history.append(
                    {
                        "question": user_question,
                        "answer": response
                    }
                )

        # DISPLAY CHAT HISTORY
        if st.session_state.chat_history:

            st.subheader("🧠 Conversation History")

            for chat in reversed(st.session_state.chat_history):

                st.markdown(
                    f'''
                    <div class="user-box">
                    <b>👨 You:</b><br>{chat["question"]}
                    </div>
                    ''',
                    unsafe_allow_html=True
                )

                st.markdown(
                    f'''
                    <div class="ai-box">
                    <b>🤖 AI:</b><br>{chat["answer"]}
                    </div>
                    ''',
                    unsafe_allow_html=True
                )

        st.markdown('</div>', unsafe_allow_html=True)

    # RIGHT SIDE
    with col2:

        st.markdown('<div class="card">', unsafe_allow_html=True)

        st.subheader("📊 PDF Analytics")

        st.metric(
            "Total Words",
            len(text.split())
        )

        st.metric(
            "Characters",
            len(text)
        )

        st.metric(
            "Text Chunks",
            len(chunks)
        )

        st.metric(
            "Questions Asked",
            len(st.session_state.chat_history)
        )

        st.markdown('</div>', unsafe_allow_html=True)

    # FULL PDF TEXT
    with st.expander("📄 View Extracted PDF Text"):

        st.write(text)

# FOOTER
st.markdown("---")

st.markdown(
    '<div class="footer">Built with ❤️ using Streamlit, LangChain, FAISS, and Ollama</div>',
    unsafe_allow_html=True
)