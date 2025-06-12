import logging
import asyncio
from typing import Dict, AsyncGenerator, Any
from dataclasses import dataclass

# Imports del paquete openai-agents instalado
from agents import Runner, Agent

# Import de tu archivo local renombrado  
from my_agents import AGENTS, orchestrator_agent

from context import SyntheticDataContext, create_context

logger = logging.getLogger(__name__)

@dataclass
class StreamEvent:
    type: str
    data: Any

# Memoria conversacional simple
AGENT_CONTEXTS: dict[str, Dict[str, Any]] = {}

async def handle_message(
    message: str,
    user_id: str,
    session_id: str,
    context_store: Dict[str, SyntheticDataContext]
) -> Dict[str, str]:
    try:
        # Contexto propio del sistema
        if session_id not in context_store:
            context = create_context(user_id, session_id)
            context.current_agent = "Orchestrator"
            context_store[session_id] = context

        # Contexto para el SDK
        if session_id not in AGENT_CONTEXTS:
            AGENT_CONTEXTS[session_id] = {"context": context_store[session_id]}

        synthetic_context = context_store[session_id]
        agent_context = AGENT_CONTEXTS[session_id]

        current_agent_name = synthetic_context.current_agent or "Orchestrator"
        agent = AGENTS.get(current_agent_name, orchestrator_agent)

        # Ejecutar el mensaje
        result = await Runner.run(agent, input=message, context=agent_context)

        # Cambios de agente (handoff) - versión corregida para openai-agents
        max_handoffs = 3  # Evitar loops infinitos
        handoff_count = 0

        # En openai-agents, los handoffs se manejan automáticamente por el Runner
        # Solo necesitamos actualizar el agente actual si hay un cambio
        if hasattr(result, 'last_agent') and result.last_agent:
            synthetic_context.current_agent = result.last_agent.name
        else:
            # Mantener el agente actual si no hay cambios
            synthetic_context.current_agent = agent.name

        return {
            "session_id": session_id,
            "agent": synthetic_context.current_agent,
            "response": result.final_output
        }

    except Exception as e:
        logger.error(f"Error en handle_message: {e}", exc_info=True)
        # Fix para el error de context_store
        current_agent = "unknown"
        if session_id in context_store and hasattr(context_store[session_id], 'current_agent'):
            current_agent = context_store[session_id].current_agent or "unknown"
        
        return {
            "session_id": session_id,
            "agent": current_agent,
            "response": f"❌ Error interno: {str(e)}"
        }

async def handle_message_stream(
    message: str,
    user_id: str,
    session_id: str,
    context_store: Dict[str, SyntheticDataContext]
) -> AsyncGenerator[StreamEvent, None]:
    try:
        if session_id not in context_store:
            context = create_context(user_id, session_id)
            context.current_agent = "Orchestrator"
            context_store[session_id] = context

        if session_id not in AGENT_CONTEXTS:
            AGENT_CONTEXTS[session_id] = {"context": context_store[session_id]}

        synthetic_context = context_store[session_id]
        agent_context = AGENT_CONTEXTS[session_id]

        current_agent_name = synthetic_context.current_agent or "Orchestrator"
        agent = AGENTS.get(current_agent_name, orchestrator_agent)

        # Para streaming, procesar normalmente y simular eventos
        result = await Runner.run(agent, input=message, context=agent_context)

        # Simular eventos de streaming
        yield StreamEvent(type="message_start", data={"agent": agent.name})
        
        # Dividir la respuesta en chunks
        response = result.final_output
        chunk_size = 50
        for i in range(0, len(response), chunk_size):
            chunk = response[i:i+chunk_size]
            yield StreamEvent(type="content_block_delta", data={"delta": {"text": chunk}})
            await asyncio.sleep(0.1)  # Simular delay
        
        yield StreamEvent(type="message_done", data={"agent": agent.name})

        # Cambiar el agente si hubo handoff
        if result.handoff:
            synthetic_context.current_agent = result.handoff
            logger.info(f"[{session_id}] Cambio a agente: {result.handoff}")

    except Exception as e:
        logger.error(f"Error en handle_message_stream: {e}", exc_info=True)
        yield StreamEvent(
            type="error_event",
            data=f"❌ Error interno: {str(e)}"
        )