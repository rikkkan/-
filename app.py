# app.py
import gradio as gr
from qa_engine import PsyQAEngine

print("正在启动 AI 心理咨询室后台服务...")

# 实例化qa_engine.py 里写的引擎类
# 这一步会把模型加载进显卡，只会执行一次！
engine = PsyQAEngine()

def chat_interface(user_message, history):
    # 直接调用引擎的推理方法
    return engine.generate_reply(user_message, history)

# 构建精简版 Gradio UI
demo = gr.ChatInterface(
    fn=chat_interface,
    title="🧠 AI 心理咨询室 (基于本地 RAG x Qwen)",
    description="我是一个私有化部署的心理咨询助手，你的所有倾诉都将在本地服务器受到严格的隐私保护。请告诉我，你遇到了什么烦心事？",
    theme="soft"
)

if __name__ == "__main__":
    print("Web 服务已就绪！")
    # 启动应用，server_name="0.0.0.0" 允许所有外部网络访问该端口
    demo.launch(share=True, server_name="0.0.0.0")
