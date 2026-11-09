from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

load_dotenv()

PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Answer only using this context. If not in context, say you don't know.\n\n"
     "Context:\n{context}"),
    ("human", "{question}"),
])


def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


def get_llm():
    return ChatGroq(
        model="openai/gpt-oss-120b",  # <-- Change this line
        temperature=0,
    )
   

def build_rag_chain(retriever):
    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | PROMPT
        | get_llm()
        | StrOutputParser()
    )