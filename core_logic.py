import os
from PyPDF2 import PdfReader

# --- LangChain 核心组件 ---
from langchain_openai import ChatOpenAI
# 🔴 修改点 1: 引入 HuggingFace 本地 Embedding
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def extract_text_from_pdf(pdf_file):
    """提取 PDF 文本"""
    text = ""
    try:
        pdf_reader = PdfReader(pdf_file)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    except Exception as e:
        return f"Error reading PDF: {e}"
    return text

def create_vectorstore(text, openai_api_key=None, base_url=None):
    """
    创建向量库 (使用本地免费模型)
    """
    if not text:
        return "PDF 内容为空，无法分析。"

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # 本地模型上下文较小，切片切小一点
        chunk_overlap=50,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    # 🔴 修改点 2: 使用本地 HuggingFace 模型
    # 这样就不需要 API Key 了，也不用担心 Base URL 配置错误
    # 模型会自动下载到本地，第一次运行比较慢，请耐心等待
    try:
        print("正在加载本地 Embedding 模型...")
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        print("正在向量化...")
        vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
        return vectorstore
    except Exception as e:
        return f"Local Embedding Error: {str(e)}"

def ask_pdf(vectorstore, question, openai_api_key, base_url=None):
    """
    RAG 问答 (这一步仍然需要 API Key 来生成回答)
    """
    if not vectorstore:
        return "请先上传文件并等待分析完成。"

    # 这里仍然使用你的 Key 进行对话
    llm = ChatOpenAI(
        temperature=0.3,
        openai_api_key=openai_api_key,
        base_url=base_url,
        model_name="gpt-3.5-turbo"
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    template = """你是一个专业的学术助手。请根据下面的【上下文】内容来回答用户的【问题】。
    
    【上下文】：
    {context}

    【问题】：
    {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain.invoke(question)
