from aiohttp.web_middlewares import middleware
# 基础设定

from pydantic.dataclasses import dataclass
@dataclass
class Context:
    user_id:str

from pydantic.dataclasses import dataclass
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

tools = [get_user_location,get_weather_for_localtion]

SYSTEM_PROMPT="""你是一个生活助手。
你有两个工具可以使用：
- get_weather_for_location：使用此工具可获得当地天气
- get_user_location：使用此工具可以获得用户所在的城市

如果用户询问天气，请确定你知道用户所在的城市。无论用户问题的城市在哪，都使用get_user_location来获取用户所在城市。
"""

# 静态模型
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
base_model = ChatOpenAI(
    model="qwen-plus-2025-12-01",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature=0.1,
    max_tokens=1000,
    timeout=30
)
# agent = create_agent(
#     model=base_model,
#     tools=tools,
#     system_prompt=SYSTEM_PROMPT
# )

# 动态模型（模型中间件）
from langchain.agents.middleware import wrap_model_call,ModelRequest,ModelResponse

advanced_model = ChatOpenAI(
    model="qwen3-max-2026-01-23",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

@wrap_model_call
def dynamic_model_selection(request:ModelResponse,handler)->ModelResponse:
    """基于对话的复杂度选择模型"""
    message_count = len(request.state["messages"])

    if message_count>10:
        # 长对话使用高级模型
        model = advanced_model
    else:
        model = base_model

    request.model = model
    return handler(request)

# agent = create_agent(
#     model=base_model,
#     tools=tools,
#     middleware=[dynamic_model_selection],
#     context_schema=Context,
# )
#
# agent.invoke(
#     {"messages":[{"role":"user","content":"外面天气怎么样"}]},
#     context=Context(user_id="01")
# )

# 工具错误处理（工具中间件）

from langchain.agents.middleware import wrap_tool_call
from langchain_core.messages import ToolMessage

@wrap_tool_call
def handle_tool_errors(request,handler):
    """使用自定义消息处理工具处理错误"""
    try:
        return handler(request)
    except Exception as e:
        # 返回自定义错误信息
        return ToolMessage(
            content=f"Tool error:Please check your import and try again.({str(e)})",
            tool_call_id=request.tool_call["id"]
        )

agent = create_agent(
    model=base_model,
    tools=tools,
    middleware=[handle_tool_errors]
)

