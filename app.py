import streamlit as st
import time
from core_logic import extract_text_from_pdf, summarize_paper

# 1. 页面配置 (Page Config)
st.set_page_config(
    page_title="AI 论文速读助手",
    page_icon="📚",
    layout="wide"
)

# 2. 侧边栏 (Sidebar) - 用于文件上传和介绍
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3022/3022340.png", width=80)
    st.title("📚 Paper Reader")
    st.markdown("---")

    # 文件上传组件
    uploaded_file = st.file_uploader(
        "请上传 PDF 论文",
        type=["pdf"],
        help="建议上传 20 页以内的学术论文"
    )

    st.markdown("---")
    st.markdown("""
    ### 使用说明
    1. 上传 PDF 文件
    2. 系统自动解析文本
    3. 点击按钮生成摘要

    *Powered by DeepSeek & LangChain*
    """)

# 3. 主界面 (Main Area)
st.header("🤖 AI 智能论文摘要生成器")

if uploaded_file is not None:
    # --- 阶段 A: 解析 PDF ---
    with st.status("正在解析 PDF...", expanded=True) as status:
        st.write("正在读取二进制数据...")
        # 调用我们在 core_logic 中写的函数
        text_content = extract_text_from_pdf(uploaded_file)

        # 简单的校验
        if len(text_content) < 100:
            st.error("⚠️ 无法识别 PDF 中的文字！这可能是扫描件（图片 PDF）。请上传文字版 PDF。")
            st.stop()  # 停止后续运行

        st.write(f"✅ 解析成功！共提取字符数：{len(text_content)}")

        # 显示前 500 个字符预览，让用户放心
        with st.expander("点击预览提取的原始文本"):
            st.text(text_content[:1000] + "...")

        status.update(label="PDF 解析完成", state="complete", expanded=False)

    # --- 阶段 B: 调用 AI ---
    # 添加一个按钮，避免一上传就开始扣费/调用
    if st.button("🚀 开始生成摘要", type="primary"):

        # 显示进度条/加载动画
        with st.spinner('正在请求 DeepSeek 大脑，请稍候... (通常需要 10-30 秒)'):
            try:
                start_time = time.time()

                # 调用核心逻辑
                summary = summarize_paper(text_content)

                end_time = time.time()

                # --- 阶段 C: 结果展示 ---
                st.success(f"生成完毕！耗时 {end_time - start_time:.2f} 秒")
                st.markdown("### 📝 论文摘要")
                st.markdown("---")

                # 将结果渲染为 Markdown
                st.markdown(summary)

                # 额外的贴心功能：下载摘要
                st.download_button(
                    label="💾 下载摘要为 Markdown",
                    data=summary,
                    file_name="summary.md",
                    mime="text/markdown"
                )

            except Exception as e:
                st.error(f"发生错误: {e}")
                st.warning("请检查你的 API Key 是否正确，或网络是否通畅。")

else:
    # 引导页面 (当没有上传文件时显示)
    st.info("👈 请先在左侧侧边栏上传一个 PDF 文件。")

    # 放一些装饰性的内容
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="支持格式", value="PDF")
    with col2:
        st.metric(label="模型驱动", value="DeepSeek-V3")
    with col3:
        st.metric(label="开发耗时", value="< 1 Hour")