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
from langchain_core.documents.base import Document
from langchain.document_loaders import PyPDFLoader


# Custom TextFileLoader
class TextFileLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load(self):
        with open(self.file_path, 'r', encoding='utf-16') as file:
            content = file.read()
            doc = Document(page_content=content, metadata={'source':'https://luatvietnam.vn/y-te/luat-bao-hiem-y-te-2008-39053-d1.html'})
        return [doc]

# Load documents
def load_documents():
    document_path = 'documents\luat-bao-hiem-y-te.txt'
    loader = TextFileLoader(
        file_path=document_path  # Update this path to your local text file
    )
    docs = loader.load()
    return docs

# Split documents
def split_documents(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80, separators=[
        "\n\n",
        "\n",
        " ",
        ".",
        ",",
        "\u200b",  # Zero-width space
        "\uff0c",  # Fullwidth comma
        "\u3001",  # Ideographic comma
        "\uff0e",  # Fullwidth full stop
        "\u3002",  # Ideographic full stop
        "",
    ],)
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
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, verbose=True)
        
        # Prompt
        template = """
            Bạn đang truy vấn từ cơ sở dữ liệu văn bản pháp luật Việt Nam. Dưới đây là câu hỏi cần bạn trả lời dựa trên các tài liệu pháp lý có sẵn. 
            Hãy cung cấp câu trả lời một cách chính xác và chi tiết nhất có thể, trích dẫn các điều luật hoặc quy định liên quan nếu cần thiết.
            Trước khi trả lời câu hỏi, bạn cần biết cấu trúc của 1 văn bản luật:
            1. Phần mở đầu: Bao gồm tên bộ luật, tên cơ quan hành chính ban hành, ngày tháng ban hành, ký hiệu văn bản. Phần này bạn nên chú ý thời điểm luật được ban hành. Nội dung này sẽ ít được hỏi hơn.
            2. Phần nội dung: Có bố cục như sau:
                Chương, Điều, Khoản, điểm.
                - Mỗi điểm chỉ thể hiện 1 ý và phải trình bày trong 1 câu hoặc đoạn.
                - Chương, điều đều phải có tên chỉ nội dung chính.
                - Chương sẽ bao gồm Điều. Điều bao gồm Khoản và điểm. 
                - Chương được đánh số la mã. Điều được đánh số la tinh.
                - Các khoản nằm trong điều thường có kí tự chữ cái la tinh ở đầu tiên. Ví dụ: "a)"
                - Chú ý các đoạn văn bản với chữ đầu tiên là "Điều" với một số theo sau.
            3. Kết thúc văn bản: Bao gồm:
                - Chức vụ, họ tên và chữ ký của người có thẩm quyền ban hành luật.
                - Bắt đầu bằng "Luật này đã được Quốc hội nước Cộng hòa xã hội chủ nghĩa Việt Nam khóa"

            Hướng dẫn trả lời:
            1. Đọc kĩ câu hỏi và xác định bộ luật, điều luật
            2. Trả lời ngắn gọn và chính xác nhất có thể. Đảm bảo sự dụng từ ngữ có trong văn bản.
            3. Nếu có yêu cầu trích dẫn thì thực thiện trích dẫn đặt t định về mục đích, nguyên tắc, phạm vi áp dụng và quy định cơ bản về bảo hiểm y tế.rong "". 
            
            Câu hỏi: {question}
            Câu trả lời:
        """

        prompt = ChatPromptTemplate.from_template(template)


        # Post-processing
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # Chain
        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
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

