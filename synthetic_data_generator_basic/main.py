"""
Main FastAPI application for Synthetic Data Generator - Basic Version
Simple web interface for the OpenAI agents with STREAMING support
"""

import asyncio
import uuid
import os
import sys
from conversation_handler import handle_message, handle_message_stream

async def cli_loop():
    print("🧠 Generador de Datos Sintéticos (Modo CLI con Streaming)")
    
    # Verificar si existe la API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: No se encontró OPENAI_API_KEY")
        print("💡 Crea un archivo .env con: OPENAI_API_KEY=tu_api_key_aqui")
        return
    
    user_id = "cli_user"
    session_id = str(uuid.uuid4())
    print(f"🔄 Nueva sesión: {session_id}")
    print("💬 Escribe 'exit', 'quit' o 'salir' para terminar")
    print("🌊 Modo streaming activado - verás las respuestas en tiempo real")
    
    while True:
        try:
            user_input = input("\n👤 Tú: ")
            if user_input.lower() in ["exit", "quit", "salir"]:
                print("👋 ¡Hasta luego!")
                break

            if not user_input.strip():
                continue

            print("🤔 ", end="", flush=True)
            
            # 🌊 USAR STREAMING
            current_agent = "unknown"
            response_text = ""
            
            async for event in handle_message_stream(
                message=user_input,
                user_id=user_id,
                session_id=session_id,
                context_store={}
            ):
                if event.type == "message_start":
                    current_agent = event.data.get("agent", "unknown")
                    print(f"\n🤖 {current_agent}: ", end="", flush=True)
                
                elif event.type == "content_block_delta":
                    # ✨ Mostrar texto en tiempo real
                    delta_text = event.data.get("delta", {}).get("text", "")
                    if delta_text:
                        print(delta_text, end="", flush=True)
                        response_text += delta_text
                
                elif event.type == "agent_change":
                    old_agent = event.data.get("old_agent")
                    new_agent = event.data.get("new_agent")
                    print(f"\n🔄 Handoff: {old_agent} → {new_agent}")
                    print(f"🤖 {new_agent}: ", end="", flush=True)
                    current_agent = new_agent
                
                elif event.type == "tool_call":
                    tool_name = event.data.get("tool_name")
                    print(f"\n🛠️  Llamando herramienta: {tool_name}")
                    print(f"🤖 {current_agent}: ", end="", flush=True)
                
                elif event.type == "tool_result":
                    print(f"\n✅ Herramienta completada")
                    print(f"🤖 {current_agent}: ", end="", flush=True)
                
                elif event.type == "message_done":
                    print()  # Nueva línea al final
                    final_agent = event.data.get("agent")
                    if final_agent != current_agent:
                        print(f"🏁 Completado por: {final_agent}")
                
                elif event.type == "error_event":
                    print(f"\n❌ Error: {event.data}")
                    break

        except KeyboardInterrupt:
            print("\n👋 Interrumpido por el usuario. Saliendo...")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

async def cli_loop_no_streaming():
    """Versión alternativa sin streaming para comparar"""
    print("🧠 Generador de Datos Sintéticos (Modo CLI - Sin Streaming)")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: No se encontró OPENAI_API_KEY")
        return
    
    user_id = "cli_user"
    session_id = str(uuid.uuid4())
    print(f"🔄 Nueva sesión: {session_id}")
    
    while True:
        try:
            user_input = input("\n👤 Tú: ")
            if user_input.lower() in ["exit", "quit", "salir"]:
                break

            if not user_input.strip():
                continue

            print("🤔 Procesando...")
            result = await handle_message(
                message=user_input,
                user_id=user_id,
                session_id=session_id,
                context_store={}
            )

            print(f"\n🤖 {result['agent']}: {result['response']}")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    # Cargar variables de entorno si existe .env
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        print("💡 Tip: Instala python-dotenv para cargar .env automáticamente")
        print("   pip install python-dotenv")
    
    # Comprobar argumentos
    if len(sys.argv) > 1 and sys.argv[1] == "no-stream":
        asyncio.run(cli_loop_no_streaming())
    else:
        asyncio.run(cli_loop())