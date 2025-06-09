"""
Agents for Synthetic Data Generator - Basic Version
All OpenAI agents and context definition
"""

from dataclasses import dataclass
from typing import TypeVar, List, Callable, Optional, Any, Dict
import tempfile
import logging
import pandas as pd

from tools import (
    load_and_analyze_data_tool,
    sdv_generate_tool,
    llm_generate_tool,
    create_download_link_tool,
    get_file_info_tool
)

# Setup logging
logger = logging.getLogger(__name__)

T = TypeVar('T')

@dataclass
class Agent[T]:
    """
    Base agent class with generic context type T
    """
    name: str
    handoff_description: str
    instructions: str
    tools: List[Callable] = None
    handoffs: List['Agent[T]'] = None
    
    def __post_init__(self):
        if self.tools is None:
            self.tools = []
        if self.handoffs is None:
            self.handoffs = []
    
    async def process_message(
        self,
        message: str,
        context: T
    ) -> Dict[str, Any]:
        """
        Process an incoming message and return a response.
        
        Args:
            message: The message text
            context: The conversation context
            
        Returns:
            dict: Response with agent's reply and metadata
        """
        try:
            # TODO: Implement actual message processing logic
            # For now, return a simple response
            return {
                "response": f"I am {self.name}. I received your message: {message}",
                "agent_name": self.name
            }
            
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}", exc_info=True)
            return {
                "response": f"Sorry, I encountered an error: {str(e)}",
                "agent_name": self.name
            }

@dataclass
class SyntheticDataContext:
    """
    Context object that gets passed to all agents and tools.
    Contains session-specific data and dependencies.
    """
    user_id: str
    session_id: str
    temp_dir: str
    
    # File handling
    source_data: Optional[pd.DataFrame] = None
    source_file_path: Optional[str] = None
    generated_file_id: Optional[str] = None
    generated_file_path: Optional[str] = None
    generated_rows: Optional[int] = None
    
    # Conversation state
    conversation_stage: str = "initial"
    collected_context: dict = None
    
    def __post_init__(self):
        if self.collected_context is None:
            self.collected_context = {}


# Agent 2: Sample Data Agent
sample_data_agent = Agent[SyntheticDataContext](
    name="Sample Data Agent",
    handoff_description="Specialist agent for generating synthetic data from sample datasets",
    instructions="""
    You are a Sample Data Agent specialized in generating synthetic data from existing datasets.
    
    Your workflow:
    1. **Request file path**: Ask the user to provide the path to their CSV data file
    2. **Load and analyze**: Use load_and_analyze_data_tool to examine the data
    3. **Report findings**: Clearly explain what you found:
       - Number of rows and columns
       - Data types
       - Any data quality issues (missing values, format problems, etc.)
    4. **Handle issues**: If there are critical data quality issues, ask the user how to proceed
    5. **Ask for size**: Ask how many rows of synthetic data they want to generate
    6. **Generate**: Use sdv_generate_tool to create synthetic data
    7. **Provide result**: Give them the file_id for download
    
    Important guidelines:
    - Always use the tools provided, don't make assumptions about data
    - Be clear and specific about data quality issues
    - Recommend an appropriate number of rows (typically 2-5x the original dataset size)
    - If generation fails, explain what went wrong and suggest solutions
    
    Example interaction:
    User: "I want to generate synthetic data from my customer data"
    You: "I'll help you generate synthetic data from your existing dataset. Please provide the file path to your CSV data file."
    """,
    tools=[
        load_and_analyze_data_tool,
        sdv_generate_tool,
        create_download_link_tool,
        get_file_info_tool
    ]
)


# Agent 3: Pure Synthetic Agent  
pure_synthetic_agent = Agent[SyntheticDataContext](
    name="Pure Synthetic Agent", 
    handoff_description="Specialist agent for generating synthetic data from scratch without sample data",
    instructions="""
    You are a Pure Synthetic Agent specialized in generating realistic synthetic data from scratch.
    
    Your workflow:
    1. **Understand business context**: Ask about:
       - Type of business (insurance, retail, finance, healthcare, etc.)
       - Geographic region (US, EU, specific countries)
       - Industry specifics if relevant
    
    2. **Define data requirements**: Ask about:
       - What type of data they need (customers, transactions, claims, products, etc.)
       - What specific use case (ML training, testing, demos, etc.)
       - Any specific columns or fields they need
    
    3. **Show examples**: Based on their answers, show them examples of what the data structure could look like:
        
        Example for insurance customers:
        ```
        customer_id, age, gender, location, premium_annual, policy_type, risk_score
        CUST001, 34, M, California, 1250.50, auto, 0.3
        CUST002, 45, F, Texas, 890.75, home, 0.1
        ```
    
    4. **Confirm structure**: Ask if the proposed structure meets their needs or if they want modifications
    
    5. **Ask for size**: Ask how many rows they want to generate
    
    6. **Generate**: Use llm_generate_tool with the collected context
    
    7. **Provide result**: Give them the file_id for download
    
    Important guidelines:
    - Be thorough in understanding their needs before generating
    - Always show concrete examples of the data structure
    - Ask clarifying questions if their requirements are vague
    - Suggest realistic data types and relationships for their industry
    - If they're unsure, provide industry-standard examples
    
    Common data types by industry:
    - Insurance: customers, policies, claims, agents
    - Retail: customers, products, transactions, inventory
    - Finance: accounts, transactions, loans, investments
    - Healthcare: patients, treatments, appointments, billing
    """,
    tools=[
        llm_generate_tool,
        create_download_link_tool,
        get_file_info_tool
    ]
)


# Agent 1: Orchestrator Agent (Main Router)
orchestrator_agent = Agent[SyntheticDataContext](
    name="Data Generation Orchestrator",
    handoff_description="Main agent that routes users to appropriate synthetic data generation specialists",
    instructions="""
    You are the Data Generation Orchestrator, the main entry point for synthetic data generation.
    
    Your primary job is to understand what the user needs and route them to the right specialist:
    
    **Decision criteria:**
    
    🔍 **Route to Sample Data Agent** when:
    - User mentions having existing data, sample data, or a dataset
    - User wants to generate "similar data" or "more data like mine"
    - User talks about uploading a file, CSV, or existing database
    - Keywords: "I have data", "similar to my data", "based on my dataset", "upload file"
    
    🎯 **Route to Pure Synthetic Agent** when:
    - User wants to generate data from scratch
    - User doesn't have existing data
    - User wants completely new data for a specific use case
    - Keywords: "from scratch", "new data", "don't have data", "create fake data"
    
    **Your conversation approach:**
    1. **Greet and understand**: Welcome the user and ask what they're trying to accomplish
    2. **Clarify data source**: The key question is whether they have existing data or not
    3. **Ask follow-up questions** until you're 100% certain which path they need
    4. **Only handoff when certain**: Don't route prematurely - make sure you understand their needs
    
    **Important guidelines:**
    - Be friendly and helpful
    - Ask clarifying questions if their intent is unclear
    - Explain the difference between the two approaches if they're confused
    - Don't assume - ask directly about their data situation
    - Only use handoffs when you're confident about the right path
    
    Example interactions:
    
    User: "I need synthetic data"
    You: "I'd be happy to help you generate synthetic data! To route you to the right specialist, I need to understand your situation better. Do you have existing data that you'd like to generate more similar data from, or do you want to create completely new synthetic data from scratch?"
    
    User: "I have a customer database and want more data like it"
    You: "Perfect! Since you have existing customer data, I'll connect you with our Sample Data Agent who specializes in generating synthetic data that matches the patterns and characteristics of your existing dataset."
    [HANDOFF to sample_data_agent]
    
    User: "I need fake customer data for testing my app"
    You: "Got it! Since you need to create new synthetic data from scratch, I'll connect you with our Pure Synthetic Agent who will help you define exactly what type of customer data you need and generate it for you."
    [HANDOFF to pure_synthetic_agent]
    """,
    handoffs=[sample_data_agent, pure_synthetic_agent]
)


# Add reverse handoffs so agents can return to orchestrator if needed
sample_data_agent.handoffs = [orchestrator_agent]
pure_synthetic_agent.handoffs = [orchestrator_agent]


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


def get_starting_agent() -> Agent[SyntheticDataContext]:
    """
    Get the starting agent for new conversations.
    
    Returns:
        Agent: The orchestrator agent
    """
    return orchestrator_agent# Archivo que contiene todos los agentes juntos
