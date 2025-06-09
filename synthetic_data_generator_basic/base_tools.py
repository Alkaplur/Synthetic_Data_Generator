"""
Base tools for Synthetic Data Generator - Basic Version
"""

from dataclasses import dataclass
from typing import TypeVar, Callable, Any, Dict, Optional
import logging

# Setup logging
logger = logging.getLogger(__name__)

T = TypeVar('T')

@dataclass
class RunContextWrapper[T]:
    """Wrapper for running tools with context"""
    context: T
    tool: Callable

    async def run(self, **kwargs) -> Dict[str, Any]:
        """Run the tool with context"""
        try:
            return await self.tool(context=self.context, **kwargs)
        except Exception as e:
            logger.error(f"Error running tool: {str(e)}", exc_info=True)
            return {"error": str(e)}

def function_tool[T](func: Callable) -> Callable:
    """
    Decorator to create a tool from a function.
    
    Args:
        func: The function to convert to a tool
        
    Returns:
        Callable: The tool function
    """
    async def tool_wrapper(context: T, **kwargs) -> Dict[str, Any]:
        try:
            result = await func(context, **kwargs)
            return {"success": True, "result": result}
        except Exception as e:
            logger.error(f"Error in tool {func.__name__}: {str(e)}", exc_info=True)
            return {"success": False, "error": str(e)}
    
    tool_wrapper.__name__ = func.__name__
    tool_wrapper.__doc__ = func.__doc__
    return tool_wrapper 