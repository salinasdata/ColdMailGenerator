import asyncio

from fastapi import WebSocket
from langchain import OpenAI
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders.base import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.utilities import ApifyWrapper
from langchain.vectorstores import Chroma


class Analyze:
    def __init__(self, url):
        self.apify = ApifyWrapper()
        self.url = url

    async def get_crawl_input(self) -> dict:
        return {"htmlTransformer": "extractus",
                "crawlerType": "cheerio",
                "startUrls": [
                    {"url": self.url}]
                }

    async def get_loader(self):
        crawl_input = await self.get_crawl_input()
        return self.apify.call_actor(
            actor_id="apify/website-content-crawler",
            run_input=crawl_input,
            dataset_mapping_function=lambda item: Document(
                page_content=item["text"] or "", matadata={"source": item["url"]}
            ),
        )

    async def get_docs(self):
        loader = await self.get_loader()
        return loader.load()

    async def get_chain(self, retriever):
        chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3)
        qa = RetrievalQA.from_chain_type(llm=chat, chain_type="stuff",
                                         retriever=retriever)
        return qa

    async def logger(self, message: str, websocket: WebSocket):
        print(message)
        await websocket.send_text(message)
        await asyncio.sleep(0.01)

    async def get_email(self, websocket: WebSocket):
        template_prompt = """
                Formulate a short 5-8 line email to the website owner pitching lead 
                generation for his business. The email should 
                make a reference to his work and give him a compliment.
            """

        await self.logger("Collecting docs...", websocket)
        docs = await self.get_docs()
        await self.logger("Docs are collected!", websocket)
        await self.logger("Collecting chunks...", websocket)
        # Split the long document into smaller chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        documents = text_splitter.split_documents(docs)
        await self.logger("Chunks are collected!", websocket)
        await self.logger("Select which embeddings to use..", websocket)
        embeddings = OpenAIEmbeddings()
        db = Chroma.from_documents(documents, embeddings)
        retriever = db.as_retriever(search_type='similarity', searhc_kwargs={"k": 2})

        await self.logger("Get chain...", websocket)

        chain = await self.get_chain(retriever)
        await self.logger("Chain generated!", websocket)

        await self.logger("Generating email...", websocket)
        email = chain({"query": template_prompt})
        await self.logger("", websocket)

        await self.logger(f"<pre style='white-space: pre-wrap;'>"
                          f"{email.get('result')}</pre>",
                          websocket)
        db.delete()
        await self.logger("", websocket)
