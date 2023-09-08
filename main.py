import spacy.cli.download
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.text_splitter import SpacyTextSplitter
from langchain.vectorstores import Chroma

TOK_MODEL = "de_core_news_sm"
PATH_PDF = "./pdf"
PATH_DB = "./db"
PATH_MODEL = "./Models/openbuddy-llama2-70b-v10.1-q3_k.gguf"

# Load PDF

loader = PyPDFDirectoryLoader(PATH_PDF)
documents = loader.load()

# Tokenization
try:
    tokenizer = SpacyTextSplitter(
        pipeline=TOK_MODEL,
        chunk_size=500,
        chunk_overlap=50,
        separator='\n'
    )
except OSError:
    spacy.cli.download(TOK_MODEL)

split_doc = tokenizer.split_documents(documents)


# Embedding
embedding = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-small",
    model_kwargs={
        'device': 'cuda'
    },
    encode_kwargs={
        'normalize_embeddings': True
    }
)

# Vectorstore
store = Chroma.from_documents(
    collection_name='BobStore',
    documents=split_doc,
    embedding=embedding,
    persist_directory=PATH_DB
)

# Retriever
retriever = store.as_retriever(search_kwargs={"k": 5})

# Chain

template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer. 
Use three sentences maximum and keep the answer as concise as possible. 
Answer in German.
{context}
Question: {question}
Helpful Answer:"""

chain_prompt = PromptTemplate.from_template(template)

llm = LlamaCpp(
    model_path=PATH_MODEL,
    max_tokens=2048,
    verbose=True,
    n_ctx=2048,
    n_gpu_layers=22,
    temperature=0.1
)

retrieval_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={
        'prompt': chain_prompt
    }
)

question = "Auf welcher Höhe sollte sich ein Handlauf befinden?"
result = retrieval_chain({'query': question})
print(result["result"])
