import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langserve import add_routes

load_dotenv()

prompt_temp = ChatPromptTemplate(
    [
        ("system","你是一个全方位IT技术助手，会认真且有理有据地回答用户的问题"),
        ("human","{text}")
    ]
)

model = ChatOpenAI(
    model="glm-4-9b-0414",
    base_url="http://192.168.1.127:5000/v1"
)

parser = StrOutputParser()

chain = prompt_temp | model | parser


app = FastAPI(
    title="LangServe",
    version="1.0"
)

add_routes(
    app=app,
    runnable=chain,
    path="/chain"
)

if __name__=="__main__":
    uvicorn.run(app,host="0.0.0.0",port=8000)


