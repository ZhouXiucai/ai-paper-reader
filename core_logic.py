import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. 加载环境变量 (读取 .env)
load_dotenv()

# 获取配置
api_key = os.getenv("DEEPSEEK_API_KEY")
base_url = os.getenv("DEEPSEEK_BASE_URL")
model_name = os.getenv("DEEPSEEK_MODEL")

if not api_key:
    raise ValueError("请在 .env 文件中配置 DEEPSEEK_API_KEY")

# 2. 初始化大模型
# DeepSeek 兼容 OpenAI 格式，所以直接用 ChatOpenAI 类
llm = ChatOpenAI(
    model=model_name,
    api_key=api_key,
    base_url=base_url,
    temperature=0.3, # 温度低一点，让回答更严谨
)

def extract_text_from_pdf(uploaded_file):
    """
    从 Streamlit 上传的文件对象中提取文本
    """
    try:
        pdf_reader = PdfReader(uploaded_file)
        text = ""
        # 遍历每一页提取文本
        for page in pdf_reader.pages:
            content = page.extract_text()
            if content:
                text += content + "\n"
        return text
    except Exception as e:
        return f"Error reading PDF: {e}"

def summarize_paper(text_content):
    """
    将文本发送给 DeepSeek 进行总结
    """
    # 定义提示词模板 (Prompt Engineering)
    # 这里的技巧是：明确角色，明确输出格式，明确语言
    system_template = """
    你是一位专业的学术研究助手。你的任务是帮助用户快速理解论文的核心内容。
    请阅读用户提供的论文文本，并按照以下 Markdown 格式生成一份中文摘要：

    ## 📄 论文标题与核心贡献
    （用一句话概括这篇论文解决了什么问题）

    ## 💡 关键创新点
    - （列出3-5个关键点，使用列表形式）

    ## 🧪 方法与结论
    （简要描述使用了什么方法，得到了什么结论）

    ---
    注意：请忽略参考文献部分，只关注正文。直接输出结果，不要废话。
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("user", "{text}")
    ])

    # 构建处理链：Prompt -> LLM -> StringParser
    chain = prompt | llm | StrOutputParser()

    # 调用模型
    # 既然是 MVP，我们先直接把全文塞进去 (DeepSeek 支持长文本)
    # 如果文本超长，这里可能会报错，但对于一般论文(10-20页)没问题
    try:
        result = chain.invoke({"text": text_content})
        return result
    except Exception as e:
        return f"AI 处理出错: {e}"

# --- 本地测试代码 (开发阶段调试用) ---
if __name__ == "__main__":
    # 你可以在这里写死一个本地 PDF 路径来测试逻辑是否通顺
    # 临时测试代码
    #  with open("基于机器学习的气体传感器设计与优化_张钰.pdf", "rb") as f:
    #      text = extract_text_from_pdf(f)
    #      print(f"提取字符数: {len(text)}")
    #      print("正在发送给 AI...")
    #      res = summarize_paper(text[:5000]) # 先只测前5000字省钱省时间
    #      print(res)
    print("Core logic loaded successfully. Waiting for app.py to call.")