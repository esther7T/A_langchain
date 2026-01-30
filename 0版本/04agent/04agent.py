# 代理

# 搜索引擎调用
# Tavily是一个搜索引擎，设置了TAVILY_API_KEY
from langchain_community.tools.tavily_search import TavilySearchResults

search = TavilySearchResults(max_results=2)
# search_results = search.invoke("我的心略大于整个宇宙 全文")
# print(search_results)

tools = [search]

# 利用语言模型调用工具
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model="glm-4-9b-0414",
    base_url="http://192.168.1.127:5000/v1"
)

# -----------------------
from langchain_core.chat_history import (BaseChatMessageHistory,InMemoryChatMessageHistory)
store = {}
def get_session_history(session_id:str)->BaseChatMessageHistory:
    if session_id not in store:
        store[session_id]=InMemoryChatMessageHistory()
    return store[session_id]
# config = {"configurable":{"session_id":"agent01"}}
from langchain_core.runnables.history import RunnableWithMessageHistory
# ----------------
from langchain_core.messages import HumanMessage

# model_with_tools = model.bind_tools(tools)

# # 语言模型绑定工具后正常问答，不调用工具
# response = RunnableWithMessageHistory(model_with_tools,get_session_history).invoke(
#     [HumanMessage(content="你好呀")],
#     config = config
# )
#
# print(f"Content:{response.content}")
# print(f"ToolCalls:{response.tool_calls}")

# # 语言模型绑定工具后，调用工具-模型推荐调用某工具，但并未调用
# response = RunnableWithMessageHistory(model_with_tools,get_session_history).invoke(
#     [HumanMessage(content="今天青岛的天气怎么样")],
#     config = config
# )
#
# print(f"Content:{response.content}")
# print(f"ToolCalls:{response.tool_calls}")

# 创建代理使模型能够调用工具-0.3版本
from langchain.agents import create_agent

model_with_tool = model.bind_tools(tools)
from langgraph.prebuilt import create_react_agent
agent = create_react_agent(
    model_with_tool,  # 使用绑定后的模型
    tools=tools,
)
response = agent.invoke(
    {"messages":[HumanMessage(content="今天青岛的天气如何")]}
)
print(response["messages"])
# 此处不能正常调用因为模型部署设置问题

# # 流式token
# for step,metadata in agent.stream(
#     {"messages":[HumanMessage(content="今天青岛的天气如何")]},
#     stream_mode="messages"
# ):
#     if metadata["langgraph_node"] == "agent" and (text:=step.text()):
#         print(text ,end=" |")