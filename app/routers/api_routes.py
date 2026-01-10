"""
Chat API Routes for BabyJay
============================
Feedback handling moved to app/api/feedback.py
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
from app.db.db_client import get_db, DatabaseClient
from app.db.auth import get_current_user, get_optional_user, AuthUser
from app.rag.chat import BabyJayChat

router = APIRouter(prefix="/api", tags=["chat"])

# Request/Response models
class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    conversation_id: Optional[str]
    title: Optional[str]

class ConversationResponse(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str

class MessageResponse(BaseModel):
    id: str
    role: str
    content: str
    created_at: str

class ConversationDetailResponse(BaseModel):
    conversation: ConversationResponse
    messages: List[MessageResponse]

class UpdateTitleRequest(BaseModel):
    title: str


# ==================== CHAT ====================

@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    user: AuthUser = Depends(get_current_user),
    db: DatabaseClient = Depends(get_db)
):
    """Send a message and get a response (authenticated)."""
    conversation_id = request.conversation_id
    is_new_conversation = False
    
    # Create new conversation if needed
    if not conversation_id:
        title = db.generate_title_from_message(request.message)
        conversation = db.create_conversation(user.id, title)
        if not conversation:
            raise HTTPException(status_code=500, detail="Failed to create conversation")
        conversation_id = conversation["id"]
        is_new_conversation = True
    else:
        conversation = db.get_conversation(conversation_id, user.id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
    
    # Save user message
    db.add_message(conversation_id, "user", request.message)
    
    # Get conversation history for context
    recent_messages = db.get_recent_messages(conversation_id, limit=20)
    
    # Create chat instance
    chat_instance = BabyJayChat(
        session_id=conversation_id,
        use_redis=False,
        debug=False
    )
    
    # Load history
    for msg in recent_messages[:-1]:
        chat_instance._conversation_history.append({
            "role": msg["role"],
            "content": msg["content"]
        })
    
    # Get response
    response = chat_instance.ask(request.message, use_history=True)
    
    # Save assistant response
    db.add_message(conversation_id, "assistant", response)
    
    # Get title
    if is_new_conversation:
        title = db.generate_title_from_message(request.message)
        db.update_conversation_title(conversation_id, user.id, title)
    else:
        title = conversation["title"]
    
    return ChatResponse(
        response=response,
        conversation_id=conversation_id,
        title=title
    )


@router.post("/chat/anonymous", response_model=ChatResponse)
async def chat_anonymous(request: ChatRequest):
    """Chat without authentication (no history saved)."""
    chat_instance = BabyJayChat(
        use_redis=False,
        debug=False
    )
    
    response = chat_instance.ask(request.message, use_history=False)
    
    return ChatResponse(
        response=response,
        conversation_id=None,
        title=None
    )


# ==================== CONVERSATIONS ====================

@router.get("/conversations", response_model=List[ConversationResponse])
async def list_conversations(
    user: AuthUser = Depends(get_current_user),
    db: DatabaseClient = Depends(get_db),
    limit: int = 50
):
    """Get all conversations for the current user."""
    conversations = db.get_conversations(user.id, limit=limit)
    
    return [
        ConversationResponse(
            id=c["id"],
            title=c["title"],
            created_at=c["created_at"],
            updated_at=c["updated_at"]
        )
        for c in conversations
    ]


@router.get("/conversations/{conversation_id}", response_model=ConversationDetailResponse)
async def get_conversation(
    conversation_id: str,
    user: AuthUser = Depends(get_current_user),
    db: DatabaseClient = Depends(get_db)
):
    """Get a conversation with all its messages."""
    conversation = db.get_conversation(conversation_id, user.id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    messages = db.get_messages(conversation_id)
    
    return ConversationDetailResponse(
        conversation=ConversationResponse(
            id=conversation["id"],
            title=conversation["title"],
            created_at=conversation["created_at"],
            updated_at=conversation["updated_at"]
        ),
        messages=[
            MessageResponse(
                id=m["id"],
                role=m["role"],
                content=m["content"],
                created_at=m["created_at"]
            )
            for m in messages
        ]
    )


@router.put("/conversations/{conversation_id}")
async def update_conversation(
    conversation_id: str,
    request: UpdateTitleRequest,
    user: AuthUser = Depends(get_current_user),
    db: DatabaseClient = Depends(get_db)
):
    """Update conversation title."""
    result = db.update_conversation_title(conversation_id, user.id, request.title)
    if not result:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    return {"success": True, "title": request.title}


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    user: AuthUser = Depends(get_current_user),
    db: DatabaseClient = Depends(get_db)
):
    """Delete a conversation and all its messages."""
    success = db.delete_conversation(conversation_id, user.id)
    if not success:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    return {"success": True}


# ==================== HEALTH CHECK ====================

@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "babyjay-api"}