import asyncio

from fastapi import WebSocket
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders.base import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.utilities import ApifyWrapper


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

    async def get_chain(self):
        chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3)
        return load_qa_chain(chat, chain_type="map_reduce")

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
        await self.logger("Get chain...", websocket)

        chain = await self.get_chain()
        await self.logger("Chain generated!", websocket)

        await self.logger("Generating email...", websocket)

        email = chain.run(input_documents=documents, question=template_prompt)
        await self.logger("", websocket)

        await self.logger(f"<pre style='white-space: pre-wrap;'>{email}</pre>",
                          websocket)
        await self.logger("", websocket)
