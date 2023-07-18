import asyncio

from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders.base import Document
from langchain.utilities import ApifyWrapper
from fastapi import WebSocket


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
        return load_qa_chain(chat, chain_type="stuff")

    @staticmethod
    async def create_sentences(segments, MIN_WORDS, MAX_WORDS):
        # Combine the non-sentences together
        sentences = []

        is_new_sentence = True
        sentence_length = 0
        sentence_num = 0
        sentence_segments = []

        for i in range(len(segments)):
            if is_new_sentence == True:
                is_new_sentence = False
            # Append the segment
            sentence_segments.append(segments[i])
            segment_words = segments[i].split(' ')
            sentence_length += len(segment_words)

            # If exceed MAX_WORDS, then stop at the end of the segment
            # Only consider it a sentence if the length is at least MIN_WORDS
            if (sentence_length >= MIN_WORDS and segments[i][
                -1] == '.') or sentence_length >= MAX_WORDS:
                sentence = ' '.join(sentence_segments)
                sentences.append({
                    'sentence_num': sentence_num,
                    'text': sentence,
                    'sentence_length': sentence_length
                })
                # Reset
                is_new_sentence = True
                sentence_length = 0
                sentence_segments = []
                sentence_num += 1

        return sentences

    @staticmethod
    async def get_chunks(docs: list) -> list:
        """
        Function to break a large text into chunks
        """
        chunks_list = []
        for doc in docs:
            segments = doc.page_content.split('.')
            segments = [segment + '.' for segment in segments]
            # Further split by comma
            segments = [segment.split(',') for segment in segments]
            # Flatten
            segments = [item for sublist in segments for item in sublist]
            sentences = await Analyze.create_sentences(segments, MIN_WORDS=20,
                                                       MAX_WORDS=80)
            CHUNK_LENGTH = 5
            STRIDE = 1
            for i in range(0, len(sentences), (CHUNK_LENGTH - STRIDE)):
                chunk = ' '.join(
                    [item['text'] for item in sentences[i:i + CHUNK_LENGTH]])
                chunks_list.append(Document(page_content=chunk, metadata=doc.metadata))
        return chunks_list

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
        chunks = await self.get_chunks(docs)
        await self.logger("Chunks are collected!", websocket)
        await self.logger("Get chain...", websocket)

        chain = await self.get_chain()
        await self.logger("Chain generated!", websocket)

        await self.logger("Generating email...", websocket)

        email = chain.run(input_documents=chunks, question=template_prompt)
        await self.logger("", websocket)

        await self.logger(f"<pre style='white-space: pre-wrap;'>{email}</pre>",
                          websocket)
        await self.logger("", websocket)
