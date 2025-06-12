import logging
import asyncio
from typing import Dict, AsyncGenerator, Any
from dataclasses import dataclass

# Imports del paquete openai-agents instalado
from agents import Runner, Agent
from openai.types.responses import ResponseTextDeltaEvent

# Import de tu archivo local renombrado  
from my_agents import AGENTS, orchestrator_agent

# Import del contexto del SDK
from tools import SyntheticDataContext

logger = logging.getLogger(__name__)

@dataclass
class StreamEvent:
    type: str
    data: Any

# Almacén de contextos por sesión
SESSION_CONTEXTS: Dict[str, SyntheticDataContext] = {}

async def handle_message(
    message: str,
    user_id: str,
    session_id: str,
    context_store: Dict[str, Any]  # Ya no se usa, mantenido por compatibilidad
) -> Dict[str, str]:
    try:
        # 🎯 CREAR/OBTENER CONTEXTO DEL SDK
        if session_id not in SESSION_CONTEXTS:
            SESSION_CONTEXTS[session_id] = SyntheticDataContext(
                user_id=user_id,
                session_id=session_id
            )
        
        sdk_context = SESSION_CONTEXTS[session_id]
        current_agent_name = getattr(sdk_context, 'current_agent', 'Orchestrator')
        agent = AGENTS.get(current_agent_name, orchestrator_agent)

        # Ejecutar el mensaje con contexto del SDK
        result = await Runner.run(agent, input=message, context=sdk_context)

        # Actualizar agente actual
        if hasattr(result, 'last_agent') and result.last_agent:
            sdk_context.current_agent = result.last_agent.name
        else:
            sdk_context.current_agent = agent.name

        return {
            "session_id": session_id,
            "agent": sdk_context.current_agent,
            "response": result.final_output
        }

    except Exception as e:
        logger.error(f"Error en handle_message: {e}", exc_info=True)
        current_agent = "unknown"
        if session_id in SESSION_CONTEXTS:
            current_agent = getattr(SESSION_CONTEXTS[session_id], 'current_agent', 'unknown')
        
        return {
            "session_id": session_id,
            "agent": current_agent,
            "response": f"❌ Error interno: {str(e)}"
        }

async def handle_message_stream(
    message: str,
    user_id: str,
    session_id: str,
    context_store: Dict[str, Any]  # Ya no se usa, mantenido por compatibilidad
) -> AsyncGenerator[StreamEvent, None]:
    try:
        # 🎯 CREAR/OBTENER CONTEXTO DEL SDK
        if session_id not in SESSION_CONTEXTS:
            SESSION_CONTEXTS[session_id] = SyntheticDataContext(
                user_id=user_id,
                session_id=session_id
            )
        
        sdk_context = SESSION_CONTEXTS[session_id]
        current_agent_name = getattr(sdk_context, 'current_agent', 'Orchestrator')
        agent = AGENTS.get(current_agent_name, orchestrator_agent)

        logger.info(f"[{session_id}] Iniciando streaming con agente: {current_agent_name}")

        # ✨ STREAMING REAL usando openai-agents con contexto del SDK
        result = Runner.run_streamed(agent, input=message, context=sdk_context)
        
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
                
                # Actualizar agente actual en contexto
                sdk_context.current_agent = new_agent_name
                current_agent_name = new_agent_name

            # 🛠️ Eventos de herramientas (versión mejorada)
            elif event.type == "run_item_stream_event":
                if event.item.type == "tool_call_item":
                    # Obtener nombre de herramienta de forma segura
                    tool_name = "herramienta_desconocida"
                    tool_args = {}
                    
                    # Intentar diferentes formas de obtener el nombre
                    if hasattr(event.item, 'name'):
                        tool_name = event.item.name
                    elif hasattr(event.item, 'function') and hasattr(event.item.function, 'name'):
                        tool_name = event.item.function.name
                    elif hasattr(event.item, 'tool_name'):
                        tool_name = event.item.tool_name
                    
                    # Intentar obtener argumentos
                    if hasattr(event.item, 'args'):
                        tool_args = event.item.args
                    elif hasattr(event.item, 'function') and hasattr(event.item.function, 'arguments'):
                        tool_args = event.item.function.arguments
                    elif hasattr(event.item, 'arguments'):
                        tool_args = event.item.arguments
                    
                    yield StreamEvent(
                        type="tool_call",
                        data={
                            "tool_name": tool_name,
                            "tool_args": tool_args
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
                "agent": sdk_context.current_agent,
                "final_output": result.final_output if hasattr(result, 'final_output') and result.final_output else "Respuesta completada",
                "session_id": session_id
            }
        )

        logger.info(f"[{session_id}] Streaming completado. Agente final: {sdk_context.current_agent}")

    except Exception as e:
        logger.error(f"Error en handle_message_stream: {e}", exc_info=True)
        yield StreamEvent(
            type="error_event",
            data=f"❌ Error interno: {str(e)}"
        )