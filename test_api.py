import uvicorn
import asyncio
import json
from fastapi import FastAPI, Request
from sse_starlette.sse import EventSourceResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/events")
async def get_events(request: Request):
    async def event_stream():
        count = 0
        while True:
            if (count > 3):
                break
            # 生成事件并发送给客户端
            yield f"data: Message {count}\n\n"
            count += 1
            await asyncio.sleep(1)

    json_post = await request.json()
    data = json.loads(json.dumps(json_post))
    prompt = data.get('prompt', '')
    history = data.get('history', [])
    print(prompt)
    print(history)
    return EventSourceResponse(event_stream())

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
