import streamlit as st
import os
from core_logic import extract_text_from_pdf, create_vectorstore, ask_pdf

# 页面设置
st.set_page_config(page_title="AI Paper Chat", page_icon="📚")
st.title("📚 AI 论文阅读助手 (RAG 版)")

# 获取 API Key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    api_key = st.text_input("请输入 OpenAI API Key", type="password")

if not api_key:
    st.info("请输入 API Key 以继续。")
    st.stop()

# --- 侧边栏：文件上传 ---
with st.sidebar:
    st.header("1. 上传论文")
    uploaded_file = st.file_uploader("选择 PDF 文件", type="pdf")
    
    # 添加一个清除按钮，用于重置
    if st.button("清除/重置"):
        if "vectorstore" in st.session_state:
            del st.session_state.vectorstore
        if "messages" in st.session_state:
            del st.session_state.messages
        st.rerun()

# --- 初始化 Session State ---
# 用于保存聊天记录
if "messages" not in st.session_state:
    st.session_state.messages = []

# 用于保存向量库 (这是关键，避免重复计算)
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# --- 处理文件上传与向量化 ---
if uploaded_file and st.session_state.vectorstore is None:
    with st.spinner("正在分析论文，构建知识库... (可能需要几秒钟)"):
        # 1. 提取文本
        text = extract_text_from_pdf(uploaded_file)
        # 2. 构建向量库
        vectorstore = create_vectorstore(text, api_key)
        # 3. 存入 Session
        st.session_state.vectorstore = vectorstore
        st.success("论文分析完成！现在可以在右侧提问了。")

# --- 主界面：聊天窗口 ---
st.subheader("2. 与论文对话")

# 显示历史消息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 处理用户输入
if prompt := st.chat_input("这篇论文的主要贡献是什么？"):
    # 1. 显示用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. 生成回答
    if st.session_state.vectorstore:
        with st.chat_message("assistant"):
            with st.spinner("AI 正在思考..."):
                response = ask_pdf(st.session_state.vectorstore, prompt, api_key)
                st.markdown(response)
        
        # 3. 保存 AI 回复
        st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        st.error("请先上传 PDF 文件！")
