# 向量存储和检索器
import asyncio

from importlib_metadata import metadata
from langchain_core.documents import Document

documents = [
    Document(
        page_content="小明喜欢小红",
        metadata={"source":"test1"}
    ),
    Document(
        page_content="小红只喜欢学习",
        metadata={"source": "test1"}
    ),
    Document(
        page_content="从前有座山，山里有座庙，庙里有个老和尚在跟小和尚讲故事，讲的什么呢？讲的是一个和尚挑水喝，两个和尚抬水喝，三个和尚没水喝的故事。",
        metadata={"source": "story"}
    ),
    Document(
        page_content="不想上班不想上班不想上班，重要的事情说三遍！",
        metadata={"source": "hhh"}
    ),
    Document(
        page_content="小芳学习很好，所以小红喜欢和小芳一起玩",
        metadata={"source": "test1"}
    ),
]



from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

embedding=OpenAIEmbeddings(
        model="bge-m3",
        openai_api_base="http://192.168.1.127:5000/v1",
        chunk_size=16,
        max_retries=3,
    )

vectorstore = Chroma.from_documents(
    documents,
    embedding=embedding,
)
# 根据与字符串查询的相似性返回文档
# # 同步
# answer = vectorstore.similarity_search("小红")
# print(answer)

# # 异步
# async def test() -> None:
#     answer = await vectorstore.asimilarity_search("上班")
#     print(answer)
#
# asyncio.run(test())

# # 返回分数
# score = vectorstore.similarity_search_with_score("小明")
# print(score)

# # 根据与嵌入查询的相似性返回文档
# embed = embedding.embed_query("和尚")
# answer = vectorstore.similarity_search_by_vector(embed)
# print(answer)


# ---------------------------------------

# 检索器

from langchain_core.runnables import RunnableLambda

# 直接查询
# retriever = RunnableLambda(vectorstore.similarity_search).bind(k=1)
# answer =retriever.batch(["小红","小芳"])
# print(answer)

# 生成一个检索器进行检索，自动处理向量存储的数据库区别 VectorStoreRetriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k":2},
)
answer =retriever.batch(["小红","小芳"])
print(answer)

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="glm-4-9b-0414",
    base_url="http://192.168.1.127:5000/v1"
)

message = """
仅根据给出上下文的内容回答问题：
问题：{question}
上下文：{context}
"""

prompt = ChatPromptTemplate.from_messages([("human",message)])
rag_chain = {"context":retriever,"question":RunnablePassthrough()} | prompt | llm

response = rag_chain.invoke("小红和哪些人有关系")
print(response.content)