"""
Conversation handler for Synthetic Data Generator - Basic Version
Manages conversation state and routing between agents
"""

import os
import logging
import uuid
from typing import Optional, Dict, Any
from dataclasses import dataclass

from agents import Agent, get_starting_agent, create_context

# Setup logging
logger = logging.getLogger(__name__)

@dataclass
class Session:
    """Session data structure"""
    user_id: str
    session_id: str
    context: Any
    current_agent: Agent
    messages: list

class ConversationHandler:
    """Manages conversation state and routing"""
    
    def __init__(self):
        self.active_sessions: Dict[str, Session] = {}
    
    def get_or_create_session(self, user_id: str, session_id: Optional[str] = None) -> Session:
        """Get existing session or create new one"""
        if session_id and session_id in self.active_sessions:
            return self.active_sessions[session_id]
            
        new_session_id = session_id or str(uuid.uuid4())
        context = create_context(user_id, new_session_id)
        current_agent = get_starting_agent()
        
        session = Session(
            user_id=user_id,
            session_id=new_session_id,
            context=context,
            current_agent=current_agent,
            messages=[]
        )
        
        self.active_sessions[new_session_id] = session
        return session
    
    def get_session_info(self, session_id: str) -> dict:
        """Get information about a session"""
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
            
        session = self.active_sessions[session_id]
        return {
            "session_id": session.session_id,
            "user_id": session.user_id,
            "current_agent": session.current_agent.name,
            "message_count": len(session.messages)
        }
    
    def cleanup_session(self, session_id: str) -> bool:
        """Clean up a session and its files"""
        if session_id not in self.active_sessions:
            return False
            
        session = self.active_sessions[session_id]
        
        # Clean up temporary files
        if hasattr(session.context, 'temp_dir'):
            try:
                import shutil
                shutil.rmtree(session.context.temp_dir)
            except Exception as e:
                logger.error(f"Error cleaning up temp dir: {e}")
        
        del self.active_sessions[session_id]
        return True
    
    def list_active_sessions(self) -> list:
        """List all active sessions"""
        return [
            {
                "session_id": session_id,
                "user_id": session.user_id,
                "current_agent": session.current_agent.name,
                "message_count": len(session.messages)
            }
            for session_id, session in self.active_sessions.items()
        ]

# Global conversation handler instance
conversation_handler = ConversationHandler()

async def handle_message(
    user_id: str,
    message: str,
    session_id: Optional[str] = None,
    file_path: Optional[str] = None
) -> dict:
    """
    Handle an incoming message and route it to the appropriate agent.
    
    Args:
        user_id: Unique identifier for the user
        message: The message text
        session_id: Optional session identifier
        file_path: Optional path to uploaded file
        
    Returns:
        dict: Response with agent's reply and metadata
    """
    try:
        # Get or create session
        session = conversation_handler.get_or_create_session(user_id, session_id)
        
        # Update context if file was uploaded
        if file_path:
            session.context.source_file_path = file_path
        
        # Process message with current agent
        response = await session.current_agent.process_message(
            message=message,
            context=session.context
        )
        
        # Update session state
        session.messages.append({
            "role": "user",
            "content": message
        })
        session.messages.append({
            "role": "assistant",
            "content": response["response"]
        })
        
        # Check if agent wants to handoff
        if "handoff" in response:
            session.current_agent = response["handoff"]
            response["agent_name"] = session.current_agent.name
        
        return {
            "success": True,
            "session_id": session.session_id,
            "response": response["response"],
            "agent_name": response.get("agent_name"),
            "generated_file_id": response.get("generated_file_id"),
            "has_generated_file": bool(response.get("generated_file_id"))
        }
        
    except Exception as e:
        logger.error(f"Error handling message: {str(e)}", exc_info=True)
        return {
            "success": False,
            "session_id": session_id,
            "response": None,
            "error": str(e)
        }

def get_download_file_path(session_id: str, file_id: str) -> Optional[str]:
    """
    Get the path to a generated file.
    
    Args:
        session_id: Session identifier
        file_id: File identifier
        
    Returns:
        Optional[str]: Path to the file if it exists
    """
    if session_id not in conversation_handler.active_sessions:
        return None
        
    session = conversation_handler.active_sessions[session_id]
    
    if (hasattr(session.context, 'generated_file_id') and 
        session.context.generated_file_id == file_id and
        hasattr(session.context, 'generated_file_path')):
        return session.context.generated_file_path
        
    return None