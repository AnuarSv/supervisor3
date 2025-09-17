import os
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import List

app = FastAPI(title="Live Chat & Dashboard")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


from transformers import pipeline
print("Загрузка обученной модели...")
classifier = pipeline("text-classification", model="./final_model", device=-1)
print("Модель загружена.")
def predict(text: str) -> dict:
    return classifier(text)[0]

messages = []

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast_dict(self, data: dict):
        for connection in self.active_connections:
            await connection.send_json(data)

manager = ConnectionManager()

@app.get("/", response_class=HTMLResponse)
async def get_student_page(request: Request):
    return templates.TemplateResponse("student.html", {"request": request})

@app.get("/dashboard", response_class=HTMLResponse)
async def get_supervisor_page(request: Request):
    return templates.TemplateResponse("supervisor.html", {"request": request, "messages": messages})

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            text = await websocket.receive_text()
            
            prediction = predict(text)
            
            message_data = {
                "text": text,
                "label": prediction['label'],
                "score": f"{prediction['score']:.2%}"
            }
            
            messages.append(message_data)
            
            await manager.broadcast_dict(message_data)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
