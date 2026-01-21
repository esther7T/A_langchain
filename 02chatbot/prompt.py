import dotenv
from langchain_core.messages import HumanMessage

dotenv.load_dotenv("02.env")

from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model="glm-4-9b-0414",
    base_url="http://192.168.1.127:5000/v1"
)


from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一个生活助手，会回答所有的生活类问题。"
        ),
        MessagesPlaceholder(variable_name="messages")
    ]
)

chain = prompt|model

response = chain.invoke({"messages":[HumanMessage(content="你好呀")]})
response.content