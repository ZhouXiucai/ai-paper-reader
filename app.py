import streamlit as st
import os
from core_logic import extract_text_from_pdf, create_vectorstore, ask_pdf

st.set_page_config(page_title="AI Paper Chat", page_icon="📚")
st.title("📚 AI 论文阅读助手 (RAG 版)")

# --- 侧边栏配置 ---
with st.sidebar:
    st.header("1. 配置")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        api_key = st.text_input("OpenAI API Key", type="password")

    base_url = st.text_input("Base URL (可选, 中转/代理必填)", placeholder="例如: https://api.openai-proxy.com/v1")
    if not base_url:
        base_url = None

    st.divider()
    
    st.header("2. 上传论文")
    uploaded_file = st.file_uploader("选择 PDF 文件", type="pdf")
    
    if st.button("清除/重置"):
        if "vectorstore" in st.session_state:
            del st.session_state.vectorstore
        if "messages" in st.session_state:
            del st.session_state.messages
        st.rerun()

if not api_key:
    st.info("👈 请在左侧输入 API Key 以继续。")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# --- 处理文件 ---
if uploaded_file and st.session_state.vectorstore is None:
    with st.spinner("正在分析论文，构建知识库..."):
        text = extract_text_from_pdf(uploaded_file)
        
        # 调用核心逻辑
        result = create_vectorstore(text, api_key, base_url)
        
        # 判断结果类型
        if isinstance(result, str) and result.startswith("Embedding Error"):
            st.error("🔴 分析失败，详细错误如下：")
            st.code(result, language="text")
            st.warning("👉 请截图此错误发给 AI 架构师，或检查你的 Base URL 是否正确。")
        elif isinstance(result, str):
            st.error(result) # 其他文本错误
        else:
            # 成功返回了对象
            st.session_state.vectorstore = result
            st.success("✅ 论文分析完成！")

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
                try:
                    response = ask_pdf(st.session_state.vectorstore, prompt, api_key, base_url)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"生成回答时出错: {e}")
    else:
        st.error("请先上传 PDF 文件，并确保分析成功！")
