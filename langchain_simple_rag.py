import os

from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import TextLoader
from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict

llm = init_chat_model("gpt-4o-mini", model_provider="openai")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

vector_store = InMemoryVectorStore(embeddings)
docs = TextLoader("job_listings.txt").load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=10)
all_splits = text_splitter.split_documents(docs)

# Index chunks
_ = vector_store.add_documents(documents=all_splits)

# Define prompt for question-answering
prompt = PromptTemplate.from_template(
    """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say you don't know — don't try to make up an answer.

{context}

Question: {question}
Helpful Answer:"""
)


# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


# Define application steps
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    # This print statement is only for debugging
    # print(retrieved_docs)
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}


# Compile application and test
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

user_input = input("enter your question")
response = graph.invoke({"question": user_input})
print(response["answer"])

'''

llm = OpenAIEmbeddings(api_key=OPENAI_API_KEY)




db=Chroma.from_documents(chunks,llm)

text = input("Enter the text:")
embedding_vector = llm.embed_query(text)
docs = db.similarity_search_by_vector(embedding_vector)
print(docs)
'''


