import os
from PyPDF2 import PdfReader

# --- 关键修改：全部使用新版引入路径 ---
# 1. 模型：从 langchain_openai 引入，而不是 langchain.chat_models
from langchain_openai import ChatOpenAI

# 2. 文本分割：从 langchain_text_splitters 引入
from langchain_text_splitters import CharacterTextSplitter

# 3. Prompt 和 Document：从 langchain_core 引入
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

def extract_text_from_pdf(pdf_file):
    """
    使用 PyPDF2 从上传的 PDF 文件对象中提取文本
    """
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

def summarize_paper(text, openai_api_key):
    """
    核心逻辑：
    1. 文本切分 (防止超过 Token 限制)
    2. Map-Reduce (如果有多个片段) 或 直接总结
    这里为了 MVP 简化，我们截取前 10000 字符进行总结（省钱且快），
    如果需要处理全文，可以使用 MapReduceChain (会消耗更多 Token)。
    """
    if not text:
        return "未能提取到文本。"

    # 1. 实例化 LLM
    # temperature=0 表示结果要严谨，不要瞎编
    llm = ChatOpenAI(
        temperature=0,
        openai_api_key=openai_api_key,
        model_name="gpt-3.5-turbo" # 或者 gpt-4o-mini
    )

    # 2. 简单的截断策略 (MVP 方案)
    # 真正的论文可能很长，这里为了演示，我们只取前 12000 个字符
    # 约等于 3000-4000 tokens，正好在 GPT-3.5 的范围内
    max_chars = 12000
    text_chunk = text[:max_chars]
    
    # 如果文本太短，提示用户
    if len(text) < 100:
        return "文本内容过少，无法生成摘要。"

    # 3. 定义 Prompt
    template = """
    你是一位专业的科研论文助手。请阅读以下论文片段，并用中文输出一份结构化的摘要。
    
    要求：
    1. 用**加粗标题**列出核心观点。
    2. 总结论文的研究背景、方法和主要结论。
    3. 语言通俗易懂，适合初学者阅读。
    
    论文片段：
    {text}
    
    摘要：
    """
    
    prompt = PromptTemplate(
        input_variables=["text"],
        template=template
    )

    # 4. 调用 Chain (新版写法)
    # 使用 | 运算符构建链：Prompt -> LLM
    chain = prompt | llm
    
    try:
        response = chain.invoke({"text": text_chunk})
        # response 是一个 AIMessage 对象，我们需要它的 .content 属性
        return response.content
    except Exception as e:
        return f"调用 AI 时出错: {str(e)}"
