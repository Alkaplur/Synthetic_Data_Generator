# Generador de Datos Sintéticos

Este proyecto proporciona una herramienta para generar datos sintéticos utilizando técnicas avanzadas de generación y análisis de datos, integrando SDV (Synthetic Data Vault) y LangChain.

## Características

- 🎯 Generación de datos desde definiciones personalizadas usando LangChain
- 📊 Generación de datos desde muestras existentes usando SDV
- 🔍 Análisis automático de patrones en los datos
- 📈 Visualización de datos generados
- 💾 Exportación en múltiples formatos (CSV, JSON, Excel)
- �� Sistema de agentes coordinados que procesan las solicitudes
- 🔄 Integración con SDV para datos basados en muestras
- 🧠 Integración con LangChain para generación desde definiciones

## Instalación

1. Clona el repositorio:
```bash
git clone https://github.com/tu-usuario/synthetic-data-generator.git
cd synthetic-data-generator
```

2. Crea un entorno virtual e instala las dependencias:
```bash
conda create -n synthetic_data_env python=3.12
conda activate synthetic_data_env
pip install -r requirements.txt
```

3. Configura tu API key de OpenAI:
   - Copia el archivo `synthetic_data_generator/config.py.example` a `synthetic_data_generator/config.py`
   - Edita `config.py` y reemplaza `"tu-api-key-aqui"` con tu API key real
   - ⚠️ IMPORTANTE: El archivo `config.py` está en `.gitignore` y no se subirá al repositorio

## Uso Básico

El sistema utiliza un conjunto de agentes coordinados para procesar las solicitudes:

```python
from synthetic_data_generator.agents_system import SyntheticDataAgentSystem
import asyncio

async def main():
    # Inicializar el sistema de agentes
    agent_system = SyntheticDataAgentSystem()
    
    # Ejemplo 1: Con datos de muestra
    sample_data = {
        "empleados": [
            {"nombre": "Juan Pérez", "edad": 30, "departamento": "IT"},
            {"nombre": "Ana García", "edad": 28, "departamento": "HR"}
        ]
    }
    
    result = await agent_system.process_request(
        request="Generar 100 empleados similares",
        sample_data=sample_data
    )
    
    # Ejemplo 2: Sin datos de muestra
    result = await agent_system.process_request(
        request="Generar un catálogo de productos electrónicos con 50 items"
    )

if __name__ == "__main__":
    asyncio.run(main())
```

## Sistema de Agentes

El sistema utiliza tres agentes especializados:

1. **Agente Coordinador**: Analiza la solicitud y determina el flujo a seguir
2. **Agente Ejecutor SDV**: Procesa solicitudes con datos de muestra usando SDV
3. **Agente Definidor LangChain**: Procesa solicitudes sin datos de muestra usando LangChain

## Estructura del Proyecto

```
synthetic_data_generator/
├── core/
│   ├── __init__.py
│   ├── agent.py
│   ├── models.py
│   ├── sdv_generator.py
│   └── langchain_generator.py
├── utils/
│   ├── __init__.py
│   └── visualization.py
├── examples/
│   ├── agents_example.py
│   └── interactive_example.py
├── config.py          # Configuración (no se sube a GitHub)
└── __init__.py
```

## Componentes Principales

- **Sistema de Agentes**: Coordina el procesamiento de solicitudes a través de agentes especializados
- **Generador SDV**: Implementa la generación de datos sintéticos usando Synthetic Data Vault
- **Generador LangChain**: Proporciona capacidades de generación basadas en descripciones en lenguaje natural
- **Visualización**: Herramientas para visualizar y analizar los datos generados

## Ejemplos

Ver el directorio `examples/` para ejemplos detallados de uso:
- `agents_example.py`: Ejemplo de uso del sistema de agentes
- `interactive_example.py`: Ejemplo interactivo de generación de datos

## Contribuir

Las contribuciones son bienvenidas. Por favor, abre un issue para discutir los cambios propuestos.

## Licencia

MIT 