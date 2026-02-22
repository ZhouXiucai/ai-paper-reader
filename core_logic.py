import os
import streamlit as st # 新增：引入 streamlit 用于读取云端密钥
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv

# 加载本地 .env (本地运行时有效)
load_dotenv()

# --- 关键修改：创建一个获取密钥的函数 ---
def get_env_variable(var_name):
    """
    优先从 Streamlit Secrets 读取 (云端)，
    如果找不到，再从环境变量/本地 .env 读取。
    """
    if var_name in st.secrets:
        return st.secrets[var_name]
    return os.getenv(var_name)

# 获取配置
api_key = get_env_variable("DEEPSEEK_API_KEY")
base_url = get_env_variable("DEEPSEEK_BASE_URL")

# 初始化 LLM
# 注意：必须确保 api_key 不是 None，否则 ChatOpenAI 会报错
if not api_key:
    # 这里只是为了防止立即崩溃，实际运行时如果没有 key 会在后续调用报错
    # 但至少可以让 import 过程通过
    api_key = "missing_key" 

llm = ChatOpenAI(
    model="deepseek-chat",
    openai_api_key=api_key,       # 显式传入获取到的 Key
    openai_api_base=base_url,     # 显式传入 Base URL
    temperature=0.3,
)

def extract_text_from_pdf(pdf_file):
    """从 PDF 文件对象中提取文本"""
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

def summarize_paper(pdf_text):
    """使用 LLM 总结论文内容"""
    
    # 1. 文本分块 (防止超过 Token 限制)
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=2000, 
        chunk_overlap=200
    )
    texts = text_splitter.split_text(pdf_text)
    
    # 简单起见，我们MVP版本只取前 2 个块进行总结（省钱 + 快速）
    # 真实场景可以使用 Map-Reduce 或 Refine 模式处理全文
    input_text = " ".join(texts[:2]) 

    # 2. 定义 Prompt
    template = """
    你是一位专业的学术论文助手。请阅读以下论文片段，并用中文输出一份结构化的总结。
    
    论文片段：
    {text}
    
    请按照以下格式输出：
    1. **论文标题/核心主题**：(一句话概括)
    2. **主要解决的问题**：(痛点是什么)
    3. **核心方法/技术**：(用了什么方案)
    4. **关键结论**：(实验结果或最终发现)
    """
    
    prompt = PromptTemplate(template=template, input_variables=["text"])
    
    # 3. 创建 Chain
    chain = LLMChain(llm=llm, prompt=prompt)
    
    # 4. 执行
    result = chain.run(input_text)
    return result
