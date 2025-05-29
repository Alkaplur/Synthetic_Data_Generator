"""
Ejemplo de uso del sistema de agentes para generación de datos sintéticos
"""

import asyncio
import json
from synthetic_data_generator.agents_system import SyntheticDataAgentSystem

async def main():
    # Inicializar el sistema de agentes
    system = SyntheticDataAgentSystem()
    
    # Ejemplo 1: Generación desde datos de muestra
    print("\n=== Ejemplo 1: Generación desde datos de muestra ===")
    sample_data = [
        {
            "nombre": "Juan Pérez",
            "edad": 28,
            "salario": 55000,
            "ciudad": "Madrid",
            "departamento": "Ventas"
        },
        {
            "nombre": "Ana García",
            "edad": 32,
            "salario": 62000,
            "ciudad": "Barcelona",
            "departamento": "Marketing"
        }
    ]
    
    result1 = await system.process_request(
        "Necesito generar 100 registros de empleados similares a estos datos",
        json.dumps(sample_data)
    )
    
    if result1["success"]:
        print("\nDatos generados exitosamente:")
        print(f"Total de registros: {len(result1['data'])}")
        print("\nPrimeros 3 registros:")
        for record in result1['data'][:3]:
            print(json.dumps(record, indent=2, ensure_ascii=False))
    else:
        print(f"Error: {result1['error']}")
    
    # Ejemplo 2: Generación desde definición
    print("\n=== Ejemplo 2: Generación desde definición ===")
    result2 = await system.process_request(
        "Quiero crear datos sintéticos para un sistema de inventario de productos electrónicos"
    )
    
    if result2["success"]:
        print("\nDatos generados exitosamente:")
        print(f"Total de registros: {len(result2['data'])}")
        print("\nPrimeros 3 registros:")
        for record in result2['data'][:3]:
            print(json.dumps(record, indent=2, ensure_ascii=False))
    else:
        print(f"Error: {result2['error']}")
    
    # Ejemplo 3: Solicitud inválida
    print("\n=== Ejemplo 3: Solicitud inválida ===")
    result3 = await system.process_request(
        "Dame los datos reales de todos los usuarios de Facebook"
    )
    print(f"Resultado: {result3}")

if __name__ == "__main__":
    asyncio.run(main()) 