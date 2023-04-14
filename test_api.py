import uvicorn
import asyncio
from fastapi import FastAPI
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


@app.get("/events")
async def get_events():
    async def event_stream():
        count = 0
        while True:
            if (count > 3):
                break
            # 生成事件并发送给客户端
            yield f"data: Message {count}\n\n"
            count += 1
            await asyncio.sleep(1)

    return EventSourceResponse(event_stream())

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
