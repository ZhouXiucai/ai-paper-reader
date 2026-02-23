import streamlit as st
import os
from core_logic import extract_text_from_pdf, create_vectorstore, ask_pdf

st.set_page_config(page_title="AI Paper Chat", page_icon="📚")
st.title("📚 AI 论文阅读助手 (RAG 版)")

# --- 侧边栏配置 ---
with st.sidebar:
    st.header("1. 配置")
    
    # 1. API Key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        api_key = st.text_input("OpenAI API Key", type="password")

    # 2. Base URL (新增：为了解决 AuthenticationError)
    # 如果你是官方 Key，留空即可
    # 如果你是中转 Key，通常填 https://api.xxx.com/v1
    base_url = st.text_input("Base URL (可选, 中转/代理必填)", placeholder="例如: https://api.openai-proxy.com/v1")
    if not base_url:
        base_url = None # 确保传给后端的是 None 而不是空字符串

    st.divider()
    
    st.header("2. 上传论文")
    uploaded_file = st.file_uploader("选择 PDF 文件", type="pdf")
    
    if st.button("清除/重置"):
        if "vectorstore" in st.session_state:
            del st.session_state.vectorstore
        if "messages" in st.session_state:
            del st.session_state.messages
        st.rerun()

# 检查 Key
if not api_key:
    st.info("👈 请在左侧输入 API Key 以继续。")
    st.stop()

# --- 初始化 Session ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# --- 处理文件 ---
if uploaded_file and st.session_state.vectorstore is None:
    with st.spinner("正在分析论文，构建知识库... (如果报错，请检查 Base URL)"):
        text = extract_text_from_pdf(uploaded_file)
        # 传入 base_url
        vectorstore = create_vectorstore(text, api_key, base_url)
        
        if vectorstore:
            st.session_state.vectorstore = vectorstore
            st.success("论文分析完成！")
        else:
            st.error("分析失败！可能是 API Key 无效或 Base URL 填写错误。请检查后重试。")

# --- 聊天界面 ---
st.subheader("3. 与论文对话")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("这篇论文的主要贡献是什么？"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if st.session_state.vectorstore:
        with st.chat_message("assistant"):
            with st.spinner("AI 正在思考..."):
                # 传入 base_url
                response = ask_pdf(st.session_state.vectorstore, prompt, api_key, base_url)
                st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        st.error("请先上传 PDF 文件，并确保分析成功！")
