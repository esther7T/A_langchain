import os

from dotenv import load_dotenv
load_dotenv("/.env")   # 自动把变量注入 os.environ

# 定义模型
from langchain_openai import ChatOpenAI
model = ChatOpenAI(model="glm-4-9b-0414",
                   base_url="http://192.168.1.127:5000/v1")


# 使用模型
from langchain_core.messages import HumanMessage,SystemMessage
messages = [SystemMessage(content="你是一个全面的IT技术助手"),
    HumanMessage(content="你好")]
# model.invoke(messages)

# 定义输出解析器
from langchain_core.output_parsers import StrOutputParser
parser = StrOutputParser()

# langchain特殊语法：模型自动绑定到输出解析器
# chain = model | parser

# chain.invoke(messages)

from langchain_core.prompts import ChatPromptTemplate
sys_prompt = "将下面内容翻译为{language}"
prompt_temp = ChatPromptTemplate.from_messages(
    [("system",sys_prompt),("user","{text}")]
)

chain = prompt_temp | model | parser

result = chain.invoke({"language":"意大利语","text":"我爱你"})


