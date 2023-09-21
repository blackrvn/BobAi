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
                 temperature=0,
                 verbose=False,
                 n_ctx=4096
                 ):
        self.tokenizer = self.init_tokenizer(tokenizer)
        self.embedding = self.init_embedding()
        self.vectorstore = self.init_store(store_path, self.embedding)
        self.retriever = self.init_retriever(self.vectorstore)
        self.llm = self.init_llm(path=model_path, temperature=temperature, verbose=verbose, n_ctx=n_ctx)
        self.prompt_template = """Your task is to answer questions about the given guidelines. 
                                Do this as precise as possible. 
                                Answer in German.
                                {context}
                                Question: {question}
                                Helpful Answer:"""

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

    @staticmethod
    def init_tokenizer(tokenizer):

        def tokenizer_func():
            tokenizer_def = SpacyTextSplitter(
                pipeline=tokenizer,
                chunk_size=500,
                chunk_overlap=50,
                separator='\n'
            )
            return tokenizer_def

        try:
            tokenizer = tokenizer_func()
        except OSError:
            spacy.cli.download(tokenizer)
            tokenizer = tokenizer_func()
        finally:
            return tokenizer

    @staticmethod
    def init_retriever(store):
        return store.as_retriever(search_kwargs={"k": 3})

    @staticmethod
    def init_store(path, embedding):
        return Chroma(persist_directory=path, embedding_function=embedding)

    @staticmethod
    def init_llm(**model_kwargs):
        llm = LlamaCpp(
            model_path=model_kwargs["path"],
            verbose=model_kwargs["verbose"],
            n_ctx=model_kwargs["n_ctx"],
            n_gpu_layers=18,
            temperature=model_kwargs["temperature"]
        )
        return llm

    def set_prompt_template(self, prompt):
        self.prompt_template = prompt

    def add_docs(self, pdf_dir_path=".\\pdf\\*.pdf"):

        for pdf in glob(pdf_dir_path):
            document = self.load_pdf(pdf)
            tokenized_document = self.tokenizer.split_documents(document)
            self.vectorstore._collection.add(tokenized_document)

    def init_chain(self, question):
        chain_prompt = PromptTemplate.from_template(self.prompt_template)

        retrieval_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={
                'prompt': chain_prompt
            }
        )

        result = retrieval_chain({'query': question})
        return result
