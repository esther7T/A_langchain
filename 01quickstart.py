# 构建一个基本代理
from langchain_openai import ChatOpenAI
# model = ChatOpenAI(
#     model="glm-4.5",
#     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
#     streaming=True
# )

from langchain.agents import create_agent
# def get_weather(city:str)->str:
#     """获取给出城市的天气"""
#     return f"{city}天气一项很好"
#
# agent = create_agent(
#     model = model,
#     tools = [get_weather],
#     system_prompt="你是一个生活助手"
# )
#
# for chunk in agent.stream(
#     {"messages":[{"role":"user","content":"青岛天气怎样"}]},
#     stream_mode="updates"
# ):
#     for step,data in chunk.items():
#         print(f"step:{step}")
#         print(f"content:{data['messages'][-1].content_blocks}")
#


# -------------实战智能体--------------
# 1、定义提示词
SYSTEM_PROMPT="""你是一个生活助手。
你有两个工具可以使用：
- get_weather_for_location：使用此工具可获得当地天气
- get_user_location：使用此工具可以获得用户所在的城市

如果用户询问天气，请确定你知道用户所在的城市。无论用户问题的城市在哪，都使用get_user_location来获取用户所在城市。
"""
# 2、创建工具
from dataclasses import dataclass
from langchain.tools import tool,ToolRuntime

@tool
def get_weather_for_localtion(city:str)->str:
    """根据给出城市获取当地天气"""
    return f"{city}总是阳光灿烂"

@dataclass
class Context:
    """用户运行时上下文模板"""
    user_id:str

@tool
def get_user_location(runtime:ToolRuntime[[Context]])->str:
    """根据用户ID获取用户所在城市"""
    user_id = runtime.context.user_id
    return "青岛" if user_id =="1" else "济南"

# 3、配置模型
from langchain.chat_models import init_chat_model
model = init_chat_model(
    "qwen-plus-2025-12-01",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature=0.5,
    timeout=10,
    max_tokens=1000,
    streaming=True,
    model_provider="openai"
)

# 4、定义响应格式
from dataclasses import dataclass

@dataclass
class ResponseFormat:
    """智能体响应模板"""
    # 玩笑式回复（保底选择）
    punny_response:str
    # 天气信息
    weather_conditions:str | None = None


# 5、添加记忆
from langgraph.checkpoint.memory import InMemorySaver
checkpointer = InMemorySaver()

# 6、创建并运行agent
agent = create_agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    tools=[get_user_location,get_weather_for_localtion],
    response_format=ResponseFormat,
    checkpointer=checkpointer,
)

config = {"configurable":{"thread_id":"1"}}

for chunk in agent.stream(
    {"messages":[{"role":"user","content":"广西天气怎么样"}]},
    config=config,
    context=Context(user_id="1"),
):
    print(f"chunk:{chunk}")
