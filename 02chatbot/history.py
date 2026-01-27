import dotenv

dotenv.load_dotenv("02.env")

from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model="glm-4-9b-0414",
    base_url="http://192.168.1.127:5000/v1"
)


# ----------------------------

from langchain_core.messages import HumanMessage,AIMessage

# 最简单的单轮对话，每次提问无关上下文
# model.invoke([HumanMessage(content="你好呀")])

# 手动添加历史对话
# model.invoke(
#     [
#         HumanMessage(content="你好呀，我叫星星"),
#         AIMessage(content="你好很高兴认识你"),
#         HumanMessage(content="我叫什么")
#     ]
# )



# 手动写一个记录历史的方法，包装在消息历史中
from langchain_core.chat_history import (BaseChatMessageHistory,InMemoryChatMessageHistory)
from langchain_core.runnables.history import RunnableWithMessageHistory

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# with_message_history = RunnableWithMessageHistory(model,get_session_history)
#
# config = {"configurable": {"session_id": "abc2"}}
#
# with_message_history.invoke(
#     [HumanMessage(content="你好，我叫星星")],
#     config=config
# )
#
# with_message_history.invoke(
#     [HumanMessage(content="我叫什么")],
#     config=config
# )


# ----------------------------

# 添加提示词模板
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder

# 最简单的提示词模板运行
# prompt = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             "你是一个生活助手，会回答所有的生活类问题。"
#         ),
#         MessagesPlaceholder(variable_name="messages")
#     ]
# )
#
# chain = prompt|model
#
# with_message_history = RunnableWithMessageHistory(chain,get_session_history)
#
# config = {"configurable":{"session_id":"abc3"}}
#
# response = with_message_history.invoke(
#     [HumanMessage(content="你好，我是星")],
#     config=config
# )



# 多参数的提示词模板运行
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system","你是一个生活助手，会使用{language}回答用户的问题。"
        ),
        MessagesPlaceholder(variable_name="messages")
    ]
)

# chain = prompt | model

# config = {"configurable":{"session_id":"abc4"}}


# # 多参数直接运行
# response = chain.invoke(
#     {"messages":[HumanMessage(content="你好，我是星")],"language":"中文"}
# )
# response.content


# # 多参数封装在历史消息中
# with_message_history = RunnableWithMessageHistory(chain,get_session_history,input_messages_key='messages')
#
# response = with_message_history.invoke(
#     {"messages":[HumanMessage(content="我是星")],"language":"日文"},
#     config=config
# )
#
# response.content
#
# response = with_message_history.invoke(
#     {"messages":[HumanMessage(content="我是谁")],"language":"日文"},
#     config=config
# )
#
# response.content


# ----------------------------

# # ----自定义一个token计数器----
# import tiktoken
# encoder = tiktoken.get_encoding("cl100k_base") # 或者"p50k_base"
# def token_counter(messages)->int:
#     total = 0
#     for msg in messages:
#         total += len(encoder.encode(msg.content))
#         total += 4
#     return total

# config = {"configurable": {"session_id": "abc5"}}

# 管理对话历史
from langchain_core.messages import SystemMessage,trim_messages
from langchain_core.messages.utils import count_tokens_approximately

trimmer = trim_messages(
    max_tokens=65,
    strategy="last",
    token_counter=count_tokens_approximately,
    include_system = True,
    allow_partial = False,
    start_on="human",
)

messages = [
    SystemMessage(content="you're a good assistant"),
    HumanMessage(content="hi! I'm star"),
    AIMessage(content="hi!"),
    HumanMessage(content="I like vanilla ice cream"),
    AIMessage(content="nice"),
    HumanMessage(content="whats 2 + 2"),
    AIMessage(content="4"),
    HumanMessage(content="thanks"),
    AIMessage(content="no problem!"),
    HumanMessage(content="having fun?"),
    AIMessage(content="yes!"),
]

# trimmer.invoke(messages)

# 在链中使用
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough

chain = (
    RunnablePassthrough.assign(messages=itemgetter("messages") | trimmer)
    | prompt
    | model
)

# 直接调用链
# response = chain.invoke(
#     {
#         "messages":messages + [HumanMessage(content="what math problem did i ask")],
#         "language":"中文"
#     }
# )
#
# response.content


# 包装在消息历史中
with_messages_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="messages",
)

# response = with_messages_history.invoke(
#     {
#         "messages":messages + [HumanMessage(content="我问过你什么数学问题")],
#         "language":"中文",
#     },
#     config = config,
# )

# response.content


# ---------------------------------

# 流式处理
config = {"configurable": {"session_id": "abc6"}}
for r in with_messages_history.stream(
    {
        "messages":[HumanMessage(content="你好呀，你知道酸奶怎么做吗")],
        "language":"中文"
    },
    config = config,
):
    print(r.content,end=" | ")







