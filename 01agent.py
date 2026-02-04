from aiohttp.web_middlewares import middleware
# ----基础设定----

from pydantic.dataclasses import dataclass
@dataclass
class Context:
    user_id:str

from pydantic.dataclasses import dataclass
from langchain.tools import tool,ToolRuntime

# 定义工具
@tool
def get_weather_for_localtion(city:str)->str:
    """根据给出城市获取当地天气"""
    return f"{city}总是阳光灿烂"

@dataclass
class Context:
    """用户运行时上下文模板"""
    user_id:str

# @tool
# def get_user_location(runtime:ToolRuntime[[Context]])->str:
#     """根据用户ID获取用户所在城市"""
#     user_id = runtime.context.user_id
#     return "青岛" if user_id =="1" else "济南"
#
# tools = [get_user_location,get_weather_for_localtion]

SYSTEM_PROMPT="""你是一个生活助手。
你有两个工具可以使用：
- get_weather_for_location：使用此工具可获得当地天气
- get_user_location：使用此工具可以获得用户所在的城市

如果用户询问天气，请确定你知道用户所在的城市。无论用户问题的城市在哪，都使用get_user_location来获取用户所在城市。
"""

# ----模型----
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

# ----工具错误处理（工具中间件）----
# 工具错误处理
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

# ReAct循环中的工具使用：
# 智能体遵循ReAct模式（推理+行动），在简短的推理步骤和针对性的工具调用之间交替进行，并讲产生的观察结果反馈送到后续决策中，直到他们能够提供最终答案。

# ----系统提示----
# agent = create_agent(
#     model=base_model,
#     tools=tools,
#     middleware=[handle_tool_errors],
#     system_prompt=SYSTEM_PROMPT
# )

# 动态系统提示
from typing import TypedDict
from langchain.agents.middleware import dynamic_prompt

class ContextR(TypedDict):
    user_id:str
    user_role: str
@tool
def get_expert_talk()->str:
    """获取专家的回答"""
    return "你可是专家，什么都懂！"

@dynamic_prompt
def user_role_prompt(request:ModelRequest)->str:
    """根据用户角色生成系统提示词"""
    user_role = request.runtime.context.get("user_role","user")
    base_model="你是一个单功能助手。"
    if user_role =="expert":
        return f"{base_model}只能够利用专家工具获取专家的回答。工具：get_expert_talk"
    elif user_role =="beginner" :
        return f"{base_model}只能够利用天气工具根据用户所在地点获取天气。工具：get_weather_for_localtion"
    return base_model

agent = create_agent(
    model=base_model,
    tools=[get_weather_for_localtion,get_expert_talk],
    middleware=[user_role_prompt],
    context_schema=ContextR
)

for chunk in agent.stream(
    {"messages":[{"role":"user","content":"此处天气如何"}]},
    context=ContextR(user_id="1",user_role="expert")
):
    print(chunk,end="\n")


# ----高级概念---
# # 结构化输出：response_format

# # 工具策略：ToolStrategy-结构化输出参数,通过人工设定ToolStrategy(输出结构类)来生成结构化输出。适用于任何支持工具调用的模型。
# from pydantic import BaseModel
# from langchain.agents import create_agent
# from langchain.agents.structured_output import ToolStrategy
# class ContactInfo(BaseModel):
#     name: str
#     email: str
#     phone: str
# agent = create_agent(
#     model=base_model,
#     response_format=ToolStrategy(ContactInfo)
# )

# # 提供者策略：ProviderStrategy-使用模型提供者原生的结构化输出生成。
# from langchain.agents.structured_output import  ProviderStrategy
#
# agent = create_agent(
#     model=base_model,
#     response_format=ProviderStrategy(ContactInfo)
# )

# 内存：智能体公国消息状态自动维护对话历史。
# -- 短期记忆：储存在状态中的信息
# -- 自定义状态模式：（1）通过中间件。（2）通过state_schema在create_agent上

# # （1）通过中间件定义状态：当您的自定义状态需要被特定的中间件钩子和附加到该中间件的工具访问时，请使用中间件定义自定义状态。
# from langchain.agents import AgentState
# from langchain.agents.middleware import AgentMiddleware
# class CustomState(AgentState):
#     user_pre:dict
# class CustomMiddleware(AgentMiddleware):
#     state_schema = CustomState
#     tools =[]
#     def before_model(self,state:CustomState,runtime)->dict[str,Any]|None:
#         return None
#
# agent = create_agent(
#     base_model,
#     tools=[],
#     middleware=[CustomMiddleware()]
# )
#
# result = agent.invoke({
#     "messages": [{"role": "user", "content": "I prefer technical explanations"}],
#     "user_pre": {"style": "technical", "verbosity": "detailed"},
# })
#
# # （2）使用 state_schema 参数作为定义仅在工具中使用的自定义状态的快捷方式。
# from langchain.agents import AgentState
# class CustomState(AgentState):
#     user_pre:str
# agent = create_agent(
#     base_model,tools=[],state_schema=CustomState
# )
# # The agent can now track additional state beyond messages
# result = agent.invoke({
#     "messages": [{"role": "user", "content": "I prefer technical explanations"}],
#     "user_preferences": {"style": "technical", "verbosity": "detailed"},
# })

# 中间件
# 中间件为在执行的不同阶段自定义智能体行为提供了强大的可扩展性。您可以使用中间件来
# 在模型调用前处理状态（例如，消息截断、上下文注入）
# 修改或验证模型的响应（例如，防护措施、内容过滤）
# 使用自定义逻辑处理工具执行错误
# 根据状态或上下文实现动态模型选择
# 添加自定义日志、监控或分析
# 中间件无缝集成到智能体的执行图中，允许您在关键点拦截和修改数据流，而无需更改核心智能体逻辑。


