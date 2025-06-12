"""
Main FastAPI application for Synthetic Data Generator - Basic Version
Simple web interface for the OpenAI agents
"""

import asyncio
import uuid
from conversation_handler import handle_message
from context import create_context

# Almacenamiento en memoria de contextos por sesión
context_store = {}

async def cli_loop():
    print("🧠 Generador de Datos Sintéticos (Modo CLI)")
    user_id = "cli_user"
    session_id = str(uuid.uuid4())
    print(f"🔄 Nueva sesión: {session_id}")
    
    while True:
        try:
            user_input = input("\n👤 Tú: ")
            if user_input.lower() in ["exit", "quit", "salir"]:
                print("👋 ¡Hasta luego!")
                break

            result = await handle_message(
                message=user_input,
                user_id=user_id,
                session_id=session_id,
                context_store=context_store
            )

            print(f"\n🤖 {result['agent']}: {result['response']}")

        except KeyboardInterrupt:
            print("\n👋 Interrumpido por el usuario. Saliendo...")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(cli_loop())

