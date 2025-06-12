import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

# Import del paquete openai-agents instalado
from agents import Agent

from context import SyntheticDataContext

logger = logging.getLogger(__name__)

# Agente 1: Orquestrador
orchestrator_agent = Agent(
    name="Orchestrator",
    instructions="""
Eres un orquestador de agentes que guía al usuario hacia el agente más apropiado.

Tu objetivo es decidir cuál de estos agentes es el mejor para continuar:
- "Sample": si el usuario tiene un archivo CSV.
- "Pure_Synthetic": si el usuario no tiene archivo y quiere generar datos desde cero.
- "Pure_Historical": si el usuario habla de cargar datos históricos.

⚠️ IMPORTANTE:
Cuando tengas clara la intención, NO repitas preguntas. 
Haz el **handoff automáticamente** usando el nombre exacto del agente como `handoff`.

No preguntes si quiere cambiar, simplemente cambia con seguridad si tienes indicios suficientes.

Ejemplos:
- Si dice "no tengo csv", haz handoff a "Pure_Synthetic".
- Si dice "desde cero", haz handoff a "Pure_Synthetic".
- Si menciona "históricos", haz handoff a "Pure_Historical".
- Si menciona "tengo un archivo", haz handoff a "Sample".
""",
    # Removemos tools temporalmente para evitar el error
    tools=[],
    handoffs=[]  # Se configurarán después
)

# Agente 2: Sample Data
sample_data_agent = Agent(
    name="Sample",
    instructions="""
Eres un experto en generar datos sintéticos basados en archivos CSV.
Pregunta si el usuario ya ha subido su archivo.
Una vez lo haya hecho, puedes sugerir analizarlo, mostrar columnas, y preguntar cuántas filas quiere generar.
Genera una respuesta clara, pero no inventes datos aún.
Si el usuario no tiene csv o la calidad no es buena devuelvelo al orquestador.
""",
    tools=[],
    handoffs=[]  # Se configurarán después
)

# Agente 3: Pure Synthetic
pure_synthetic_agent = Agent(
    name="Pure_Synthetic",
    instructions="""
Eres un generador de datos sintéticos desde cero.
Pregunta al usuario el contexto del negocio, el tipo de datos que necesita (clientes, pólizas, ventas...), y cuántas filas.
Sugiere una estructura de columnas basada en eso.
Puedes guiar la conversación paso a paso.
Si el usuario no tiene csv o la calidad no es buena devuelvelo al orquestador.
""",
    tools=[],
    handoffs=[]  # Se configurarán después
)

# Agente 4: Historical Data Availability
pure_historical_agent = Agent(
    name="Pure_Historical",
    instructions="""
Eres un agente que has de facilitar la carga de datos históricos en el sistema. Pregunta al usuario como quiere hacer la carga de los
datos historicos y llama a las herramientas establecidas para la carga
Si el usuario no tiene csv o la calidad no es buena devuelvelo al orquestador.
""",
    tools=[],
    handoffs=[]  # Se configurarán después
)

# Configurar handoffs después de crear todos los agentes
orchestrator_agent.handoffs = [sample_data_agent, pure_synthetic_agent, pure_historical_agent]
sample_data_agent.handoffs = [orchestrator_agent, pure_historical_agent]
pure_synthetic_agent.handoffs = [orchestrator_agent, sample_data_agent]
pure_historical_agent.handoffs = [orchestrator_agent]

# Diccionario de agentes para fácil acceso
AGENTS = {
    "Orchestrator": orchestrator_agent,
    "Sample": sample_data_agent,
    "Pure_Synthetic": pure_synthetic_agent,
    "Pure_Historical": pure_historical_agent
}