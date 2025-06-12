"""
Context for Synthetic Data Generator - Basic Version
"""

import tempfile
import logging
from dataclasses import dataclass
from typing import Optional, Any
import pandas as pd

logger = logging.getLogger(__name__)

@dataclass
@dataclass
class SyntheticDataContext:
    user_id: str
    session_id: str
    temp_dir: str
    conversation_stage: str = "initial"
    
    current_agent: Optional[str] = None

    # Extras si los usas:
    source_file_path: Optional[str] = None
    source_data: Optional[Any] = None
    generated_file_path: Optional[str] = None
    generated_file_id: Optional[str] = None



def create_context(user_id: str, session_id: str) -> SyntheticDataContext:
    """
    Factory function to create a new context for a user session.
    
    Args:
        user_id: Unique identifier for the user
        session_id: Unique identifier for this conversation session
        
    Returns:
        SyntheticDataContext: New context object
    """
    temp_dir = tempfile.mkdtemp(prefix=f"synthetic_data_{session_id}_")
    
    context = SyntheticDataContext(
        user_id=user_id,
        session_id=session_id,
        temp_dir=temp_dir,
        conversation_stage="initial"
    )
    
    logger.info(f"Created new context for user {user_id}, session {session_id}, temp_dir: {temp_dir}")
    return context