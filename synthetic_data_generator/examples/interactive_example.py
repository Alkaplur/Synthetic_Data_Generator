"""
Ejemplo interactivo del Generador de Datos Sintéticos usando bots
"""

import asyncio
import json
import pandas as pd
from tabulate import tabulate
from synthetic_data_generator.agents_system import SyntheticDataAgentSystem

async def interactive_session():
    """Sesión interactiva con el sistema de agentes"""
    system = SyntheticDataAgentSystem()
    
    print("\n=== Bienvenido al Generador de Datos Sintéticos ===")
    print("Los bots te ayudarán a generar los datos que necesitas.")
    
    while True:
        print("\n¿Qué tipo de datos te gustaría generar?")
        print("1. Datos basados en una muestra existente")
        print("2. Datos definidos desde cero")
        print("3. Salir")
        
        opcion = input("\nSelecciona una opción (1-3): ")
        
        if opcion == "3":
            print("\n¡Gracias por usar el Generador de Datos Sintéticos!")
            break
            
        elif opcion == "1":
            print("\n=== Generación desde datos de muestra ===")
            print("Por favor, proporciona algunos datos de ejemplo en formato JSON.")
            print("Ejemplo: [{\"nombre\": \"Juan\", \"edad\": 30}, {\"nombre\": \"Ana\", \"edad\": 25}]")
            
            try:
                sample_data = input("\nIngresa tus datos de muestra: ")
                # Validar que sea JSON válido
                json.loads(sample_data)
                
                print("\n¿Cuántos registros similares necesitas generar?")
                num_records = int(input("Número de registros: "))
                
                result = await system.process_request(
                    f"Necesito generar {num_records} registros similares a estos datos",
                    sample_data
                )
                
                if result["success"]:
                    print("\n¡Datos generados exitosamente!")
                    
                    # Convertir a DataFrame
                    df = pd.DataFrame(result["data"])
                    
                    # Mostrar información del DataFrame
                    print(f"\nSe generaron {len(df)} registros.")
                    print("\nPrimeras 5 filas del DataFrame:")
                    print(tabulate(df.head(), headers='keys', tablefmt='psql', showindex=False))
                    
                    # Mostrar estadísticas básicas
                    print("\nEstadísticas básicas:")
                    print(tabulate(df.describe(), headers='keys', tablefmt='psql'))
                    
                    # Guardar DataFrame
                    filename = input("\n¿Quieres guardar los datos? Ingresa el nombre del archivo (sin extensión) o presiona Enter para omitir: ")
                    if filename:
                        df.to_csv(f"{filename}.csv", index=False)
                        print(f"\nDatos guardados en {filename}.csv")
                        
                else:
                    print("\nError:", result.get("error", "Error desconocido"))
                    
            except json.JSONDecodeError:
                print("\nError: El formato JSON no es válido. Por favor, intenta de nuevo.")
            except ValueError:
                print("\nError: Por favor, ingresa un número válido de registros.")
                
        elif opcion == "2":
            print("\n=== Generación desde definición ===")
            print("Describe el tipo de datos que necesitas generar.")
            print("Ejemplo: \"Necesito datos de productos electrónicos con nombre, precio, stock y categoría\"")
            
            description = input("\nDescribe tus datos: ")
            
            result = await system.process_request(description)
            
            if result["success"]:
                print("\n¡Datos generados exitosamente!")
                
                # Convertir a DataFrame
                df = pd.DataFrame(result["data"])
                
                # Mostrar información del DataFrame
                print(f"\nSe generaron {len(df)} registros.")
                print("\nPrimeras 5 filas del DataFrame:")
                print(tabulate(df.head(), headers='keys', tablefmt='psql', showindex=False))
                
                # Mostrar estadísticas básicas
                print("\nEstadísticas básicas:")
                print(tabulate(df.describe(), headers='keys', tablefmt='psql'))
                
                # Guardar DataFrame
                filename = input("\n¿Quieres guardar los datos? Ingresa el nombre del archivo (sin extensión) o presiona Enter para omitir: ")
                if filename:
                    df.to_csv(f"{filename}.csv", index=False)
                    print(f"\nDatos guardados en {filename}.csv")
                    
            else:
                print("\nError:", result.get("error", "Error desconocido"))
                
        else:
            print("\nOpción no válida. Por favor, selecciona 1, 2 o 3.")

if __name__ == "__main__":
    asyncio.run(interactive_session()) 