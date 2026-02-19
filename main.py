import os
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import google.generativeai as genai
from typing import Dict, List, Optional
import uuid

API_KEY = ""
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash')
app = FastAPI(title="Gemini Chat API Assignment")

chat_sessions: Dict[str, List[Dict[str, str]]] = {}

class SimplePromptTemplate:
    def __init__(self, template_str: str):
        self.template_str = template_str

    def format(self, **kwargs) -> str:
        return self.template_str.format(**kwargs)

system_template = SimplePromptTemplate(
    "You are a helpful assistant. Answer the following question: {query}"
)

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User's input message")
    session_id: Optional[str] = Field(None, description="Session ID to continue conversation")

class ChatResponse(BaseModel):
    response: str
    session_id: str
    history_length: int

class HistoryResponse(BaseModel):
    session_id: str
    history: List[Dict[str, str]]

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    session_id = request.session_id
    if not session_id:
        session_id = str(uuid.uuid4())
    
    if session_id not in chat_sessions:
        chat_sessions[session_id] = []
    
    history = chat_sessions[session_id]
    formatted_prompt = system_template.format(query=request.query)

    gemini_history = []
    for msg in history:
        gemini_history.append({
            "role": msg["role"],
            "parts": [msg["content"]]
        })

    try:
        chat = model.start_chat(history=gemini_history)
        
        response = chat.send_message(formatted_prompt)
        ai_text = response.text
    except Exception as e:
        print(f"ERROR DETAILS: {e}")
        raise HTTPException(status_code=500, detail=f"Gemini API Error: {str(e)}")

    history.append({"role": "user", "content": request.query}) 
    history.append({"role": "model", "content": ai_text})

uvicorn.run(app, host="0.0.0.0", port=8000)