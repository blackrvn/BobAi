from glob import glob
from icecream import ic
import os

import spacy.cli.download
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.text_splitter import SpacyTextSplitter
from langchain.vectorstores import Chroma


class Bob:

    def __init__(self,
                 tokenizer="de_core_news_lg",
                 store_path="./db",
                 model_path="./Models/openbuddy-llama2-70b-v10.1-q3_k.gguf",
                 promt_template="""Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer. 
Use three sentences maximum and keep the answer as concise as possible. 
Answer in German.
{context}
Question: {question}
Helpful Answer:"""):
        self.promt_template = promt_template
        self.tokenizer = tokenizer
        self.store_path = store_path
        self.model_path = model_path
        if os.path.exists(self.store_path):
            self.store = Chroma(persist_directory=self.store_path, embedding_function=self.init_embedding())
        else:
            self.store = None

    @staticmethod
    def load_pdf(pdf_path):
        loader = PyPDFLoader(pdf_path)
        return loader.load()

    @staticmethod
    def init_embedding():
        embedding = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-large",
            model_kwargs={
                'device': 'cuda'
            },
            encode_kwargs={
                'normalize_embeddings': True
            }
        )
        return embedding

    def init_tokenizer(self):
        tokenizer_def = SpacyTextSplitter(
            pipeline=self.tokenizer,
            chunk_size=500,
            chunk_overlap=50,
            separator='\n'
        )
        return tokenizer_def

    def init_store(self, document):
        store = Chroma.from_documents(
            documents=document,
            embedding=self.init_embedding(),
            persist_directory=self.store_path
        )
        self.store = store

    def add_docs(self, pdf_dir_path=".\\pdf\\*.pdf"):

        try:
            tokenizer = self.init_tokenizer()
        except OSError:
            spacy.cli.download(self.tokenizer)
            tokenizer = self.init_tokenizer()

        for pdf in glob(pdf_dir_path):
            document = self.load_pdf(pdf)
            tokenized_document = tokenizer.split_documents(document)
            self.init_store(tokenized_document)

    def init_retriever(self):
        return self.store.as_retriever(search_kwargs={"k": 3})

    def init_chain(self, temperature, question):
        chain_prompt = PromptTemplate.from_template(self.promt_template)

        llm = LlamaCpp(
            model_path=self.model_path,
            verbose=False,
            n_ctx=4096,
            n_gpu_layers=18,
            temperature=temperature
        )

        retrieval_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=self.init_retriever(),
            return_source_documents=True,
            chain_type_kwargs={
                'prompt': chain_prompt
            }
        )

        result = retrieval_chain({'query': question})
        return result
