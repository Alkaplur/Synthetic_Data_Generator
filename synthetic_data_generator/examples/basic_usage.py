import os
from synthetic_data_generator import JupyterSyntheticDataAgent
from synthetic_data_generator.config import OPENAI_API_KEY, DEFAULT_NUM_RECORDS

def main():
    # Inicializar el agente con la API key desde la configuración
    if OPENAI_API_KEY == "tu-api-key-aqui":
        raise ValueError("Por favor, configura tu API key en synthetic_data_generator/config.py")
    
    agent = JupyterSyntheticDataAgent(OPENAI_API_KEY)
    
    # Ejemplo 1: Generar datos desde una definición
    print("\n📝 Ejemplo 1: Generación desde definición")
    print("=" * 50)
    
    # Definir variables de cliente
    agent.define_variables_interactive()
    
    # Definir producto
    agent.define_product_interactive()
    
    # Generar datos
    data = agent.generate_data(num_records=DEFAULT_NUM_RECORDS)
    
    # Visualizar resultados
    agent.visualize_data()
    
    # Exportar datos
    agent.export_data("clientes_sinteticos.csv")
    
    # Ejemplo 2: Generar datos desde una muestra
    print("\n📊 Ejemplo 2: Generación desde muestra")
    print("=" * 50)
    
    # Analizar datos de muestra
    agent.analyze_sample_data("datos_muestra.csv")
    
    # Generar datos
    data = agent.generate_data(num_records=DEFAULT_NUM_RECORDS)
    
    # Visualizar resultados
    agent.visualize_data()
    
    # Exportar datos
    agent.export_data("clientes_sinteticos_muestra.csv")
    
    # Mostrar resumen
    print("\n📋 Resumen del proceso")
    print("=" * 50)
    print(agent.get_summary())

if __name__ == "__main__":
    main() 