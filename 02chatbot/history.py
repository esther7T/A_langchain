import dotenv

dotenv.load_dotenv("/.env")

from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model="glm-4-9b-0414",
    base_url="http://192.168.1.127:5000/v1"
)

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


from langchain_core.chat_history import (BaseChatMessageHistory,InMemoryChatMessageHistory)
from langchain_core.runnables.history import RunnableWithMessageHistory

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

with_message_history = RunnableWithMessageHistory(model,get_session_history)

config = {"configurable": {"session_id": "abc2"}}

with_message_history.invoke(
    [HumanMessage(content="你好，我叫星星")],
    config=config
)

with_message_history.invoke(
    [HumanMessage(content="我叫什么")],
    config=config
)




