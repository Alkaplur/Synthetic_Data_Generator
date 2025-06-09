"""
Main FastAPI application for Synthetic Data Generator - Basic Version
Simple web interface for the OpenAI agents
"""

import os
import logging
import asyncio
from typing import Optional
from fastapi import FastAPI, HTTPException, Form, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from conversation_handler import handle_message, get_download_file_path, conversation_handler

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="Synthetic Data Generator",
    description="AI-powered synthetic data generation using OpenAI Agents",
    version="1.0.0"
)


class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    message: str
    user_id: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    success: bool
    session_id: str
    response: str
    agent_name: Optional[str] = None
    generated_file_id: Optional[str] = None
    has_generated_file: bool = False
    error: Optional[str] = None


@app.get("/")
async def root():
    """Root endpoint with basic info"""
    return {
        "message": "Synthetic Data Generator API",
        "version": "1.0.0",
        "endpoints": {
            "chat": "/chat",
            "chat_with_file": "/chat-with-file", 
            "download": "/download/{file_id}",
            "session_info": "/session/{session_id}",
            "health": "/health"
        }
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint for conversing with agents.
    
    Args:
        request: Chat request with message, user_id, and optional session_id
        
    Returns:
        ChatResponse with agent's response and metadata
    """
    try:
        logger.info(f"Chat request from user {request.user_id}: {request.message[:100]}...")
        
        result = await handle_message(
            user_id=request.user_id,
            message=request.message,
            session_id=request.session_id
        )
        
        return ChatResponse(**result)
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat-with-file", response_model=ChatResponse)
async def chat_with_file(
    message: str = Form(...),
    user_id: str = Form(...),
    session_id: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None)
):
    """
    Chat endpoint that supports file uploads.
    
    Args:
        message: User's message
        user_id: User identifier
        session_id: Optional session identifier
        file: Optional uploaded file
        
    Returns:
        ChatResponse with agent's response and metadata
    """
    try:
        file_path = None
        
        if file:
            # Save uploaded file temporarily
            import tempfile
            temp_dir = tempfile.mkdtemp()
            file_path = os.path.join(temp_dir, file.filename)
            
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)
            
            logger.info(f"File uploaded: {file.filename} -> {file_path}")
        
        logger.info(f"Chat with file request from user {user_id}: {message[:100]}...")
        
        result = await handle_message(
            user_id=user_id,
            message=message,
            session_id=session_id,
            file_path=file_path
        )
        
        return ChatResponse(**result)
        
    except Exception as e:
        logger.error(f"Error in chat-with-file endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/download/{file_id}")
async def download_file(file_id: str, session_id: str):
    """
    Download a generated file.
    
    Args:
        file_id: File identifier
        session_id: Session identifier
        
    Returns:
        FileResponse with the CSV file
    """
    try:
        file_path = get_download_file_path(session_id, file_id)
        
        if not file_path or not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")
        
        logger.info(f"Downloading file {file_id} from session {session_id}")
        
        return FileResponse(
            path=file_path,
            filename=f"synthetic_data_{file_id}.csv",
            media_type="text/csv"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading file {file_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/session/{session_id}")
async def get_session_info(session_id: str):
    """
    Get information about a session.
    
    Args:
        session_id: Session identifier
        
    Returns:
        Dict with session information
    """
    try:
        session_info = conversation_handler.get_session_info(session_id)
        return session_info
        
    except Exception as e:
        logger.error(f"Error getting session info for {session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/session/{session_id}")
async def cleanup_session(session_id: str):
    """
    Clean up a session and its files.
    
    Args:
        session_id: Session identifier
        
    Returns:
        Dict with cleanup result
    """
    try:
        success = conversation_handler.cleanup_session(session_id)
        
        if success:
            return {"message": f"Session {session_id} cleaned up successfully"}
        else:
            raise HTTPException(status_code=404, detail="Session not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cleaning up session {session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions")
async def list_sessions():
    """
    List all active sessions.
    
    Returns:
        Dict with all active sessions
    """
    try:
        sessions = conversation_handler.list_active_sessions()
        return {
            "active_sessions": sessions,
            "count": len(sessions)
        }
        
    except Exception as e:
        logger.error(f"Error listing sessions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        Dict with health status
    """
    return {
        "status": "healthy",
        "version": "1.0.0",
        "active_sessions": len(conversation_handler.active_sessions)
    }


# CLI mode for testing
async def cli_mode():
    """
    Simple CLI interface for testing the agents without web server.
    """
    print("🤖 Synthetic Data Generator - CLI Mode")
    print("Type 'quit' to exit\n")
    
    user_id = "cli_user"
    session_id = None
    
    while True:
        try:
            user_input = input("\n💬 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not user_input:
                continue
            
            print("🤔 Agent is thinking...")
            
            result = await handle_message(
                user_id=user_id,
                message=user_input,
                session_id=session_id
            )
            
            if result["success"]:
                session_id = result["session_id"]
                print(f"\n🤖 {result.get('agent_name', 'Agent')}: {result['response']}")
                
                if result.get("has_generated_file"):
                    print(f"\n📄 File generated: {result['generated_file_id']}")
            else:
                print(f"\n❌ Error: {result.get('error', 'Unknown error')}")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "cli":
        # Run in CLI mode
        asyncio.run(cli_mode())
    else:
        # Run FastAPI server
        import uvicorn
        uvicorn.run(
            app, 
            host="0.0.0.0", 
            port=8000, 
            log_level="info"
        )# Punto de entrada y runner del generador de datos sintéticos
