"""
Sistema de Agentes para Generación de Datos Sintéticos
Integración con SDV y LangChain
"""

import asyncio
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import json
import pandas as pd
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOpenAI

from .core.langchain_generator import (
    analyze_sample_data,
    generate_synthetic_data,
    create_schema_from_description
)
from .core.sdv_generator import SDVGenerator
from .config import OPENAI_API_KEY

# Contexto compartido entre agentes
@dataclass
class SyntheticDataContext:
    user_input: str
    sample_data: Optional[str] = None
    has_sample_data: bool = False
    data_requirements: Optional[str] = None
    generated_data: Optional[Dict[str, Any]] = None
    schema: Optional[Dict[str, Any]] = None

class BaseAgent:
    """Clase base para todos los agentes"""
    def __init__(self, name: str, instructions: str):
        self.name = name
        self.instructions = instructions
        self.next_agent = None

    async def process(self, context: SyntheticDataContext) -> Dict[str, Any]:
        raise NotImplementedError("Los agentes deben implementar este método")

class CoordinatorAgent(BaseAgent):
    """Agente coordinador que analiza necesidades y decide el flujo"""
    def __init__(self):
        super().__init__(
            name="Coordinador de Datos Sintéticos",
            instructions="""
            Eres el agente coordinador que analiza las necesidades del usuario para generación de datos sintéticos.
            
            Tu trabajo es:
            1. Entender exactamente qué tipo de datos sintéticos necesita el usuario
            2. Determinar si el usuario ya tiene datos de muestra o no
            3. Clasificar el tipo de datos y volumen requerido
            4. Decidir a qué agente hacer handoff:
               - Si el usuario TIENE datos de muestra → handoff al "Ejecutor SDV"
               - Si el usuario NO TIENE datos de muestra → handoff al "Definidor LangChain"
            """
        )

    async def process(self, context: SyntheticDataContext) -> Dict[str, Any]:
        if context.has_sample_data:
            return await self.next_agent.process(context)
        else:
            return await self.next_agent.process(context)

class SDVExecutorAgent(BaseAgent):
    """Agente ejecutor usando SDV para datos de muestra"""
    def __init__(self):
        super().__init__(
            name="Ejecutor SDV",
            instructions="""
            Eres el agente ejecutor especializado en trabajar con datos de muestra usando SDV.
            
            Tu trabajo es:
            1. Analizar la estructura de los datos de muestra
            2. Entrenar el modelo SDV
            3. Generar datos sintéticos manteniendo las correlaciones
            """
        )
        self.sdv_generator = SDVGenerator()

    async def process(self, context: SyntheticDataContext) -> Dict[str, Any]:
        try:
            # Convertir datos de muestra a DataFrame
            sample_data = pd.read_json(context.sample_data)
            
            # Entrenar modelo SDV
            train_result = self.sdv_generator.train_from_sample(sample_data)
            if not train_result["success"]:
                return train_result
            
            # Generar datos sintéticos
            generation_result = self.sdv_generator.generate_data(num_rows=100)
            if not generation_result["success"]:
                return generation_result
            
            return {
                "success": True,
                "message": "Datos generados exitosamente con SDV",
                "data": generation_result["data"].to_dict(orient='records'),
                "quality_score": generation_result["quality_score"],
                "metadata": generation_result["metadata"]
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Error en flujo SDV: {str(e)}"
            }

class LangChainDefinerAgent(BaseAgent):
    """Agente definidor usando LangChain para datos sin muestra"""
    def __init__(self):
        super().__init__(
            name="Definidor LangChain",
            instructions="""
            Eres el agente definidor que ayuda a usuarios que NO tienen datos de muestra.
            
            Tu trabajo es:
            1. Ayudar a definir la estructura de los datos
            2. Guiar en la definición de variables y restricciones
            3. Generar datos sintéticos usando LangChain
            """
        )

    async def process(self, context: SyntheticDataContext) -> Dict[str, Any]:
        try:
            # Crear schema desde la descripción
            schema_result = create_schema_from_description(context.user_input)
            if not schema_result["success"]:
                return schema_result

            # Generar datos sintéticos
            generation_result = generate_synthetic_data(
                schema_json=json.dumps(schema_result["schema"]),
                volume=100  # Volumen por defecto
            )

            if not generation_result["success"]:
                return generation_result

            return {
                "success": True,
                "message": "Datos generados exitosamente con LangChain",
                "data": generation_result["data"],
                "schema": schema_result["schema"]
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Error en flujo LangChain: {str(e)}"
            }

class SyntheticDataAgentSystem:
    def __init__(self):
        """Inicializa el sistema de agentes"""
        self.context = None
        
        # Crear y configurar los agentes
        self.coordinator = CoordinatorAgent()
        self.sdv_executor = SDVExecutorAgent()
        self.langchain_definer = LangChainDefinerAgent()
        
        # Configurar el flujo de agentes
        self.coordinator.next_agent = self.sdv_executor  # Por defecto, pero se cambiará según el caso

    async def process_request(self, user_input: str, sample_data: str = None) -> dict:
        """
        Procesa una solicitud del usuario a través del sistema de agentes
        
        Args:
            user_input: La solicitud del usuario
            sample_data: Datos de muestra opcionales (JSON string)
        """
        try:
            # Crear contexto
            self.context = SyntheticDataContext(
                user_input=user_input,
                sample_data=sample_data,
                has_sample_data=bool(sample_data)
            )

            # 1. Validar la solicitud
            if not self._validate_request(user_input):
                return {
                    "success": False,
                    "error": "Solicitud no válida para generación de datos sintéticos"
                }

            # 2. Configurar el flujo de agentes según el caso
            if self.context.has_sample_data:
                self.coordinator.next_agent = self.sdv_executor
            else:
                self.coordinator.next_agent = self.langchain_definer

            # 3. Procesar la solicitud a través de los agentes
            return await self.coordinator.process(self.context)

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "context": self.context
            }

    def _validate_request(self, user_input: str) -> bool:
        """Valida si la solicitud es apropiada para generación de datos"""
        valid_keywords = [
            "generar", "crear", "sintetizar", "datos",
            "prueba", "testing", "desarrollo", "dataset",
            "muestra", "mock", "simulación"
        ]
        
        return any(keyword in user_input.lower() for keyword in valid_keywords)

# Función de ejemplo para usar el sistema
async def main():
    """Ejemplos de uso del sistema de agentes"""
    
    system = SyntheticDataAgentSystem()
    
    print("=== EJEMPLO 1: Usuario CON datos de muestra (SDV) ===")
    sample_data = '''[
        {"nombre": "Juan Pérez", "edad": 28, "salario": 55000, "ciudad": "Madrid"},
        {"nombre": "Ana García", "edad": 32, "salario": 62000, "ciudad": "Barcelona"}
    ]'''
    
    result1 = await system.process_request(
        "Necesito generar 500 registros de empleados como estos datos",
        sample_data
    )
    print(f"Resultado: {result1}")
    
    print("\n=== EJEMPLO 2: Usuario SIN datos de muestra (LangChain) ===")
    result2 = await system.process_request(
        "Quiero crear datos sintéticos para un sistema de inventario de productos"
    )
    print(f"Resultado: {result2}")
    
    print("\n=== EJEMPLO 3: Solicitud inválida ===")
    result3 = await system.process_request(
        "Dame los datos reales de todos los usuarios de Facebook"
    )
    print(f"Resultado: {result3}")

if __name__ == "__main__":
    asyncio.run(main()) 