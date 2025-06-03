"""
Funciones de generación de datos sintéticos usando LangChain
"""

import json
from typing import Dict, Any, List
from pydantic import BaseModel, create_model
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_experimental.tabular_synthetic_data.openai import (
    create_openai_data_generator,
    OPENAI_TEMPLATE
)
from langchain_experimental.tabular_synthetic_data.prompts import (
    SYNTHETIC_FEW_SHOT_PREFIX,
    SYNTHETIC_FEW_SHOT_SUFFIX,
)
from langchain_openai import ChatOpenAI

def analyze_sample_data(sample_data_json: str) -> dict:
    """
    Analiza datos de muestra usando LangChain para extraer estructura y patrones
    
    Args:
        sample_data_json: Datos de muestra en formato JSON string
        
    Returns:
        dict: Análisis de la estructura y patrones
    """
    try:
        # Parsear datos de muestra
        sample_data = json.loads(sample_data_json)
        
        # LLM para análisis
        llm = ChatOpenAI(temperature=0, model="gpt-4")
        
        # Prompt para análisis
        analysis_prompt = PromptTemplate(
            input_variables=["data"],
            template="""
            Analiza estos datos de muestra y extrae:
            1. Nombres y tipos de columnas
            2. Rangos de valores numéricos
            3. Patrones en strings (formatos, longitudes)
            4. Relaciones entre campos
            5. Restricciones observadas
            
            Datos: {data}
            
            Responde en formato JSON con:
            {{
                "columnas": {{"nombre": "tipo"}},
                "rangos_numericos": {{"campo": {{"min": X, "max": Y}}}},
                "patrones_string": {{"campo": "patron_observado"}},
                "volumen_sugerido": numero_entero,
                "observaciones": ["observacion1", "observacion2"]
            }}
            """
        )
        
        # Ejecutar análisis
        chain = analysis_prompt | llm
        response = chain.invoke({"data": json.dumps(sample_data[:3])})  # Solo primeros 3 registros
        
        # Parsear respuesta JSON
        analysis = json.loads(response.content)
        
        return {
            "success": True,
            "analysis": analysis,
            "sample_count": len(sample_data)
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Error analizando datos: {str(e)}"
        }

def generate_synthetic_data(schema_json: str, volume: int = 100) -> dict:
    """
    Genera datos sintéticos usando LangChain SyntheticDataGenerator
    
    Args:
        schema_json: Schema en formato JSON string
        volume: Número de registros a generar
        
    Returns:
        dict: Datos sintéticos generados
    """
    try:
        # Parsear schema
        schema_info = json.loads(schema_json)
        
        # Crear modelo Pydantic dinámicamente
        fields = {}
        examples = []
        
        for field_name, field_info in schema_info.get("campos", {}).items():
            field_type = str  # Default
            if field_info.get("tipo") == "int":
                field_type = int
            elif field_info.get("tipo") == "float":
                field_type = float
            elif field_info.get("tipo") == "bool":
                field_type = bool
                
            fields[field_name] = (field_type, ...)
        
        # Crear modelo dinámico
        DynamicModel = create_model('SyntheticData', **fields)
        
        # Crear ejemplos para few-shot
        if "ejemplos" in schema_info:
            for ejemplo in schema_info["ejemplos"][:3]:  # Max 3 ejemplos
                example_str = ", ".join([f"{k}: {v}" for k, v in ejemplo.items()])
                examples.append({"example": example_str})
        else:
            # Crear ejemplo genérico
            sample_values = []
            for field_name, field_info in schema_info.get("campos", {}).items():
                if field_info.get("tipo") == "int":
                    sample_values.append(f"{field_name}: 123")
                elif field_info.get("tipo") == "float":
                    sample_values.append(f"{field_name}: 123.45")
                else:
                    sample_values.append(f"{field_name}: ejemplo_valor")
            
            examples.append({"example": ", ".join(sample_values)})
        
        # Crear template para few-shot
        prompt_template = FewShotPromptTemplate(
            prefix=SYNTHETIC_FEW_SHOT_PREFIX,
            examples=examples,
            suffix=SYNTHETIC_FEW_SHOT_SUFFIX,
            input_variables=["subject", "extra"],
            example_prompt=OPENAI_TEMPLATE,
        )
        
        # Crear generador
        synthetic_data_generator = create_openai_data_generator(
            output_schema=DynamicModel,
            llm=ChatOpenAI(temperature=1, model="gpt-4"),
            prompt=prompt_template,
        )
        
        # Generar datos
        subject = schema_info.get("tipo_datos", "datos genericos")
        extra_instructions = schema_info.get("instrucciones_extra", "Genera datos realistas y variados")
        
        synthetic_results = synthetic_data_generator.generate(
            subject=subject,
            extra=extra_instructions,
            runs=min(volume, 50)  # Límite de 50 por llamada
        )
        
        # Convertir resultados a diccionarios
        results_as_dicts = []
        for result in synthetic_results:
            if hasattr(result, 'dict'):
                results_as_dicts.append(result.dict())
            else:
                results_as_dicts.append(dict(result))
        
        # Crear un objeto de salida que cumpla con el formato esperado
        output_model = create_model('OutputModel', data=(List[Dict[str, Any]], ...))
        output = output_model(data=results_as_dicts)
        
        return {
            "success": True,
            "data": output.data,
            "count": len(output.data),
            "schema_used": schema_info,
            "metadata": {
                "generated_with": "LangChain SyntheticDataGenerator",
                "model": "gpt-4",
                "temperature": 1
            }
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Error generando datos sintéticos: {str(e)}"
        }

def create_schema_from_description(description: str) -> dict:
    """
    Convierte descripción de usuario en schema estructurado para generación
    
    Args:
        description: Descripción de los datos a generar
        
    Returns:
        dict: Schema estructurado para generación
    """
    try:
        llm = ChatOpenAI(temperature=0, model="gpt-4")
        
        schema_prompt = PromptTemplate(
            input_variables=["description"],
            template="""
            Convierte esta descripción en un schema JSON para generar datos sintéticos:
            
            Descripción: {description}
            
            Crea un schema con formato:
            {{
                "tipo_datos": "descripcion_breve",
                "campos": {{
                    "nombre_campo": {{
                        "tipo": "str/int/float/bool",
                        "descripcion": "que representa",
                        "ejemplo": "valor_ejemplo"
                    }}
                }},
                "instrucciones_extra": "instrucciones específicas para la generación",
                "ejemplos": [
                    {{"campo1": "valor1", "campo2": "valor2"}},
                    {{"campo1": "valor3", "campo2": "valor4"}}
                ]
            }}
            
            Haz los campos realistas y útiles para el caso de uso descrito.
            """
        )
        
        chain = schema_prompt | llm
        response = chain.invoke({"description": description})
        
        # Parsear respuesta JSON
        schema = json.loads(response.content)
        
        return {
            "success": True,
            "schema": schema
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Error creando schema: {str(e)}"
        } 