import os
from PyPDF2 import PdfReader

# --- LangChain 核心组件 ---
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
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

def create_vectorstore(text, openai_api_key, base_url=None):
    """
    创建向量库
    增加了 base_url 参数，用于支持中转 Key
    """
    if not text:
        return None

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    # 关键修改：如果有 base_url，就传进去；否则默认为 None (官方地址)
    # 注意：某些版本的 LangChain 使用 openai_api_base，新版使用 base_url，这里做个兼容处理
    embeddings = OpenAIEmbeddings(
        openai_api_key=openai_api_key,
        base_url=base_url 
    )

    try:
        vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
        return vectorstore
    except Exception as e:
        # 这里捕获 embedding 时的错误，通常是 Key 或网络问题
        print(f"Embedding Error: {e}")
        return None

def ask_pdf(vectorstore, question, openai_api_key, base_url=None):
    """
    RAG 问答
    """
    if not vectorstore:
        return "请先上传文件并等待分析完成。"

    # 关键修改：传入 base_url
    llm = ChatOpenAI(
        temperature=0.3,
        openai_api_key=openai_api_key,
        base_url=base_url,
        model_name="gpt-3.5-turbo"
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    template = """你是一个专业的学术助手。请根据下面的【上下文】内容来回答用户的【问题】。
    如果你在上下文中找不到答案，就诚实地说“我无法在文档中找到答案”，不要瞎编。

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
