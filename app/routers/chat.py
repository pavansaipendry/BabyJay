"""
Chat Router - API endpoints for BabyJay chat
=============================================
Provides HTTP endpoints to interact with the chatbot.

Endpoints:
    POST /chat - Send a message, get a response
    POST /chat/clear - Clear conversation history
    GET /chat/health - Check if chat system is working
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from app.rag.chat import BabyJayChat

router = APIRouter(prefix="/chat", tags=["chat"])

# Initialize chatbot (singleton)
_chat_instance: Optional[BabyJayChat] = None


def get_chat() -> BabyJayChat:
    """Get or create the chat instance"""
    global _chat_instance
    if _chat_instance is None:
        try:
            _chat_instance = BabyJayChat()
        except ValueError as e:
            raise HTTPException(status_code=500, detail=str(e))
    return _chat_instance


# Request/Response models
class ChatRequest(BaseModel):
    message: str
    use_history: bool = True

    class Config:
        json_schema_extra = {
            "example": {
                "message": "Where can I eat on campus?",
                "use_history": True
            }
        }


class ChatResponse(BaseModel):
    response: str
    success: bool = True

    class Config:
        json_schema_extra = {
            "example": {
                "response": "There are several dining options on campus...",
                "success": True
            }
        }


class HealthResponse(BaseModel):
    status: str
    documents_loaded: int


# Endpoints
@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Send a message to BabyJay and get a response.
    
    - **message**: Your question about KU (dining, transit, courses)
    - **use_history**: Whether to remember previous messages in this session
    """
    try:
        bot = get_chat()
        response = bot.ask(request.message, use_history=request.use_history)
        return ChatResponse(response=response, success=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/clear")
async def clear_history():
    """Clear the conversation history"""
    try:
        bot = get_chat()
        bot.clear_history()
        return {"message": "Conversation history cleared", "success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check if the chat system is working"""
    try:
        bot = get_chat()
        doc_count = bot.retriever.collection.count()
        return HealthResponse(status="healthy", documents_loaded=doc_count)
    except Exception as e:
        raise HTTPException(
            status_code=503, 
            detail=f"Chat system unhealthy: {str(e)}"
        )