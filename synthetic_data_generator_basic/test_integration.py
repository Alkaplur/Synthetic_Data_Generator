"""
Test de integración Pure Synthetic con agentes
"""

import asyncio
from conversation_handler import handle_message

async def test_pure_synthetic_agent():
    print("🧪 Probando agente Pure_Synthetic...")
    
    user_id = "test_user"
    session_id = "test_session"
    context_store = {}
    
    # Mensaje que debería dirigir al agente Pure_Synthetic
    message = "Quiero generar datos de clientes desde cero"
    
    try:
        result = await handle_message(
            message=message,
            user_id=user_id,
            session_id=session_id,
            context_store=context_store
        )
        
        print(f"✅ Agente usado: {result['agent']}")
        print(f"📝 Respuesta: {result['response'][:200]}...")
        
        if result['agent'] == 'Pure_Synthetic':
            print("🎉 ¡Handoff al agente Pure_Synthetic exitoso!")
            return True
        else:
            print(f"⚠️  Se esperaba Pure_Synthetic, pero se usó: {result['agent']}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(test_pure_synthetic_agent())