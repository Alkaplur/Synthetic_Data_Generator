import asyncio
from tools import SyntheticDataContext
from pure_synthetic_tools import test_ollama_connection
from agents import RunContextWrapper

async def test_connection():
    print("🧪 Probando conexión básica con Ollama...")
    
    # Crear contexto simple
    context = SyntheticDataContext(
        user_id="test",
        session_id="test"
    )
    
    # Probar conexión
    result = test_ollama_connection("nemotron-mini")
    print(f"Resultado: {result}")
    
    if result["success"]:
        print("✅ ¡Conexión exitosa!")
        return True
    else:
        print("❌ Error en la conexión")
        return False

if __name__ == "__main__":
    asyncio.run(test_connection())