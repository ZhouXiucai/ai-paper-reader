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

def create_vectorstore(text, openai_api_key):
    """
    1. 切分文本
    2. 将文本转换为向量 (Embeddings)
    3. 存入 FAISS 向量库
    """
    if not text:
        return None

    # 1. 文本切分：每块 1000 字符，重叠 200 字符（防止句子被切断）
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    # 2. 初始化 Embeddings 模型 (这是 RAG 的核心，用于理解语义)
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    # 3. 创建向量库 (运行在内存中)
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
    
    return vectorstore

def ask_pdf(vectorstore, question, openai_api_key):
    """
    RAG 核心链路：检索 -> 增强 -> 生成
    """
    if not vectorstore:
        return "请先上传文件。"

    # 1. 定义 LLM
    llm = ChatOpenAI(
        temperature=0.3, # 问答稍微有点创造性也可以，但主要还是要严谨
        openai_api_key=openai_api_key,
        model_name="gpt-3.5-turbo"
    )

    # 2. 定义检索器 (Retriever)：只找最相关的 3 个片段
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 3. 定义 Prompt
    template = """你是一个专业的学术助手。请根据下面的【上下文】内容来回答用户的【问题】。
    如果你在上下文中找不到答案，就诚实地说“我无法在文档中找到答案”，不要瞎编。

    【上下文】：
    {context}

    【问题】：
    {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    # 4. 构建 LCEL 链 (LangChain Expression Language)
    # 逻辑：retriever 找文档 -> 格式化文档 -> 填入 prompt -> 发给 llm -> 解析字符串
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # 5. 执行
    return rag_chain.invoke(question)

# 保留之前的 summarize_paper 函数，以免旧代码报错，虽然我们这次主要用 RAG
def summarize_paper(text, openai_api_key):
    # 这里为了简单，我们还是用之前的方法，或者直接调用 ask_pdf 让他生成摘要也可以
    # 为了演示兼容性，保留原逻辑的简化版
    if not text: return "无内容"
    llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)
    return llm.invoke(f"请总结以下内容：{text[:3000]}").content
