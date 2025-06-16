"""
Test super simple para verificar conexión con Ollama
"""

def test_ollama_direct():
    print("🧪 Probando conexión directa con Ollama...")
    
    try:
        from langchain_ollama import OllamaLLM
        
        # Crear conexión simple
        llm = OllamaLLM(
            model="nemotron-mini",
            base_url="http://localhost:11434",
            temperature=0.7,
        )
        
        print("✅ Import de LangChain OK")
        
        # Prueba simple
        response = llm.invoke("Hello, how are you?")
        print(f"✅ Respuesta de nemotron-mini: {response[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_ollama_direct()
    if success:
        print("🎉 ¡Conexión básica funciona!")
    else:
        print("💥 Hay problemas de configuración")