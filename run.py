import asyncio
import os
from pathlib import Path

import aiohttp
import openai
import uvicorn
from apify_client._errors import ApifyApiError
from fastapi import FastAPI, WebSocket, Request, WebSocketDisconnect
from fastapi.templating import Jinja2Templates
from openai.error import AuthenticationError

from cold_mail_generator import Analyze

BASE_DIR = Path(__file__).resolve().parent

templates = Jinja2Templates(directory=str(Path(BASE_DIR, 'templates')))
log_file = "Your Results"

port = os.environ.get("PORT") or 8000
host_url = os.environ.get("HOST") or "0.0.0.0:8000"

app = FastAPI(title='Cold Mail Generator Demo')


@app.get("/")
async def get(request: Request):
    """Log file viewer

    Args:
        request (Request): Default web request.

    Returns:
        TemplateResponse: Jinja template with context data.
    """
    context = {"title": "Cold Mail Generator Demo",
               "log_file": log_file,
               "host_url": host_url}
    return templates.TemplateResponse("index.html",
                                      {"request": request, "context": context})


@app.websocket("/ws/")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Wait for any message from the client
            params = await websocket.receive_json()
            website_url = params.get('website_url')
            openai_key = params.get('openai_key')
            apify_key = params.get('apify_key')
            is_http = website_url.strip().lower().startswith('http')
            is_url = True if website_url is not None and is_http else False
            is_openai_key = True if openai_key is not None and '-' in openai_key else False
            if is_url and is_openai_key and apify_key:
                async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(
                        verify_ssl=False)) as aio_session:
                    async with aio_session.get(website_url) as website:
                        try:
                            if website.status == 200:
                                await websocket.send_text(
                                    "Details saved. Please do not reload the page!"
                                )
                                await asyncio.sleep(0.01)
                                await websocket.send_text("Set OpenAI secret key!")
                                await asyncio.sleep(0.01)
                                openai.api_key = openai_key
                                os.environ['OPENAI_API_KEY'] = openai_key
                                await websocket.send_text("Set Apify secret key!")
                                await asyncio.sleep(0.01)
                                openai.api_key = openai_key
                                os.environ['APIFY_API_TOKEN'] = apify_key

                                await websocket.send_text("Analyzing...")
                                await asyncio.sleep(0.01)

                                analyze = Analyze(website_url)
                                await analyze.get_email(websocket)
                                await asyncio.sleep(0.01)
                                await websocket.send_text("Email is generated")

                            else:
                                await websocket.send_text("Please check provided details")
                                await asyncio.sleep(0.01)
                        except (AuthenticationError, ApifyApiError) as e:
                            await websocket.send_text(f"<pre style='white-space: "
                                                      f"pre-wrap; color:red;'"
                                                      f">{e}</pre>")
                            await asyncio.sleep(0.01)
            else:
                await websocket.send_text("Please check provided details")
                await asyncio.sleep(0.01)

            await websocket.send_text('Analyzing completed.')
            await asyncio.sleep(0.01)

    except WebSocketDisconnect as e:
        print('error:', e)
        await websocket.send_text(f'error:{e}')
        await asyncio.sleep(0.01)


if __name__ == "__main__":
    uvicorn.run(
        "run:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
        reload=True,
        workers=4,
    )
