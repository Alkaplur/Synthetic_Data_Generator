import logging
import asyncio
from typing import Dict, AsyncGenerator, Any
from dataclasses import dataclass

# Imports del paquete openai-agents instalado
from agents import Runner, Agent
from openai.types.responses import ResponseTextDeltaEvent

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

        # En openai-agents, los handoffs se manejan automáticamente
        # Solo actualizamos el agente actual
        if hasattr(result, 'last_agent') and result.last_agent:
            synthetic_context.current_agent = result.last_agent.name
        else:
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

        logger.info(f"[{session_id}] Iniciando streaming con agente: {current_agent_name}")

        # ✨ STREAMING REAL usando openai-agents
        result = Runner.run_streamed(agent, input=message, context=agent_context)
        
        # Enviar evento de inicio
        yield StreamEvent(
            type="message_start", 
            data={"agent": agent.name, "session_id": session_id}
        )

        # Stream de eventos en tiempo real
        async for event in result.stream_events():
            
            # 🔥 Eventos de texto (deltas de tokens)
            if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                if event.data.delta:
                    yield StreamEvent(
                        type="content_block_delta",
                        data={"delta": {"text": event.data.delta}}
                    )
            
            # 🤖 Cambios de agente (handoffs)
            elif event.type == "agent_updated_stream_event":
                new_agent_name = event.new_agent.name
                logger.info(f"[{session_id}] Handoff detectado: {current_agent_name} -> {new_agent_name}")
                
                yield StreamEvent(
                    type="agent_change",
                    data={
                        "old_agent": current_agent_name,
                        "new_agent": new_agent_name
                    }
                )
                
                # Actualizar agente actual
                synthetic_context.current_agent = new_agent_name
                current_agent_name = new_agent_name

            # 🛠️ Eventos de herramientas
            elif event.type == "run_item_stream_event":
                if event.item.type == "tool_call_item":
                    yield StreamEvent(
                        type="tool_call",
                        data={
                            "tool_name": event.item.name,
                            "tool_args": event.item.args
                        }
                    )
                elif event.item.type == "tool_call_output_item":
                    yield StreamEvent(
                        type="tool_result",
                        data={
                            "tool_output": event.item.output
                        }
                    )
                elif event.item.type == "message_output_item":
                    logger.info(f"[{session_id}] Mensaje completo generado")
        
        # Evento de finalización
        yield StreamEvent(
            type="message_done",
            data={
                "agent": synthetic_context.current_agent,
                "final_output": result.final_output,
                "session_id": session_id
            }
        )

        logger.info(f"[{session_id}] Streaming completado. Agente final: {synthetic_context.current_agent}")

    except Exception as e:
        logger.error(f"Error en handle_message_stream: {e}", exc_info=True)
        yield StreamEvent(
            type="error_event",
            data=f"❌ Error interno: {str(e)}"
        )