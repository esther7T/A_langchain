# 模型是代理的推理引擎。它们驱动代理的决策过程，确定调用哪些工具、如何解释结果以及何时提供最终答案。
# 初始化模型
from langchain.chat_models import init_chat_model
model = init_chat_model(
    model="qwen-plus-2025-12-01",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature=0.1,
    timeout=30,
    max_tokens=2000,
    max_retries=2,
    model_provider="openai"
)

# 调用
# # （1）直接调用：invoke
# response = model.invoke("你好呀")
# print(response)
# #  -- 字典格式消息
# conversation = [
#     {"role":"system","content":"你是一个翻译助手，能将中文翻译成日语"},
#     {"role": "user", "content": "翻译：你好，很高兴认识你"},
#     {"role": "assistant", "content": "こんにちは、はじめまして"},
#     {"role": "user", "content": "翻译：十年生死两茫茫，不思量，自难忘。千里孤坟，无处话凄凉。"},
# ]
# response = model.invoke(conversation)
# print(response)
# #  -- 消息对象
# from langchain_core.messages import HumanMessage,AIMessage,SystemMessage
# conversation = [
#     SystemMessage("你是一个翻译助手，能将中文翻译成日语"),
#     HumanMessage("翻译：你好，很高兴认识你"),
#     AIMessage("こんにちは、はじめまして"),
#     HumanMessage("翻译：纵使相逢应不识，尘满面，鬓如霜。")
# ]
# response = model.invoke(conversation)
# print(response)

# # （2）流式处理
# for chunk in model.stream("将进酒全文"):
#     print(chunk.text,end=" |",flush=True)

# # （3）批量处理：批量处理对模型的独立请求集合可以显著提高性能并降低成本，因为处理可以并行完成。
# responses = model.batch([
#     "天空为什么是蓝色的",
#     "草为什么是绿色的",
#     "我困了"
# ])
# for response in responses:
#     print(response)
#
# for response in model.batch_as_completed([
#     "天空为什么是蓝色的",
#     "草为什么是绿色的",
#     "我困了"
# ]):
#     print(response)

# 工具调用
from langchain.tools import tool
@tool
def get_weather(location:str)->str:
    """获取在某地点的天气"""
    return f"{location}阳光灿烂"

model_with_tools = model.bind_tools([get_weather])

response = model_with_tools.invoke("青岛的天气如何")

for tool_call in response.tool_calls:
    print(f"Tool:{tool_call['name']}")
    print(f"Args:{tool_call['args']}")
# （1）工具执行循环

