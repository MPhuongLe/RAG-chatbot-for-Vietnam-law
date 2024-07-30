import os
import streamlit as st
import bs4 # Beautiful soup
from langchain import hub # NLP related stuffs
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma # Similarity search and indexing
from langchain_core.output_parsers import StrOutputParser # Parse answer to desirable output (my guess is embed -> NL)
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate

# Load documents
def load_documents():
    loader = WebBaseLoader(
        web_paths=("https://luatvietnam.vn/y-te/luat-bao-hiem-y-te-2008-39053-d1.html",),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(class_=("the-document-body ndthaydoi noidungtracuu",))
        ),
    )
    docs = loader.load()
    return docs

# Split documents
def split_documents(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    return splits

# Embed documents
def embed_documents(splits):
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    return vectorstore

# Define the chat interface
def chat_interface():
    st.title("Chatbot Interface")
    docs = load_documents()
    splits = split_documents(docs)
    vectorstore = embed_documents(splits)
    st.session_state["vectorstore"] = vectorstore

    # Input area for user questions
    user_question = st.text_input("Ask a question:")
    if user_question:
        retriever = st.session_state["vectorstore"].as_retriever()

        # Define the conversational retrieval chain
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        
        # Prompt
        template = """
            Bạn đang truy vấn từ cơ sở dữ liệu văn bản pháp luật Việt Nam. Dưới đây là câu hỏi cần bạn trả lời dựa trên các tài liệu pháp lý có sẵn. Hãy cung cấp câu trả lời một cách chính xác và chi tiết nhất có thể, trích dẫn các điều luật hoặc quy định liên quan nếu cần thiết.

            Hướng dẫn trả lời:
            1. Đọc kỹ câu hỏi và xác định các từ khóa quan trọng.
            2. Tìm kiếm các văn bản pháp luật liên quan đến câu hỏi.
            3. Trích dẫn chính xác các điều luật, quy định hoặc văn bản pháp lý có liên quan.
            4. Giải thích ngắn gọn nhưng đầy đủ để người hỏi hiểu rõ về điều luật hoặc quy định đó.
            5. Nếu có thể, đưa ra ví dụ cụ thể hoặc tình huống áp dụng thực tế để minh họa.

            Câu hỏi: {question}

            Câu trả lời:
        """

        prompt = ChatPromptTemplate.from_template(template)


        # Post-processing
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # Chain
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
    
        # Get the answer
        result = rag_chain.invoke(user_question)
        st.write(result)

# Run the chat interface
if __name__ == "__main__":
    chat_interface()

