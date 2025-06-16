"""
Pure Synthetic Tools - LangChain + Nemotron Integration
Herramientas para generar datos sintéticos desde cero usando LLM
"""

import os
import logging
import pandas as pd
import json
import uuid
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from agents import function_tool, RunContextWrapper
from tools import SyntheticDataContext  # Import del contexto existente

# Imports de LangChain
#from langchain_experimental.tabular_synthetic_data import SyntheticDataGenerator
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate

logger = logging.getLogger(__name__)

# ==========================================
# CONFIGURACIÓN OLLAMA + NEMOTRON
# ==========================================

def get_ollama_llm(model_name: str = "nemotron-mini") -> Ollama:
    """
    Crear instancia de Ollama LLM
    
    Args:
        model_name: Nombre del modelo en Ollama
        
    Returns:
        Instancia de Ollama LLM configurada
    """
    try:
        llm = Ollama(
            model=model_name,
            base_url="http://localhost:11434",  # Puerto por defecto de Ollama
            temperature=0.7,  # Creatividad moderada
            num_predict=2048,  # Longitud máxima de respuesta
        )
        
        # Test básico de conectividad
        test_response = llm.invoke("Hello")
        logger.info(f"✅ Ollama conectado exitosamente con modelo: {model_name}")
        return llm
        
    except Exception as e:
        logger.error(f"❌ Error conectando con Ollama: {e}")
        raise ConnectionError(f"No se pudo conectar con Ollama. ¿Está corriendo? Error: {e}")

# ==========================================
# PLANTILLAS DE DATOS PREDEFINIDAS
# ==========================================

DATA_TEMPLATES = {
    "customers": {
        "description": "Datos de clientes/consumidores",
        "columns": [
            {"name": "customer_id", "type": "string", "description": "ID único del cliente"},
            {"name": "first_name", "type": "string", "description": "Nombre del cliente"},
            {"name": "last_name", "type": "string", "description": "Apellido del cliente"},
            {"name": "email", "type": "string", "description": "Email del cliente"},
            {"name": "age", "type": "integer", "description": "Edad del cliente (18-80)"},
            {"name": "gender", "type": "string", "description": "Género (Male, Female, Other)"},
            {"name": "city", "type": "string", "description": "Ciudad donde vive"},
            {"name": "country", "type": "string", "description": "País donde vive"},
            {"name": "registration_date", "type": "date", "description": "Fecha de registro"},
            {"name": "total_spent", "type": "float", "description": "Total gastado en euros"}
        ]
    },
    "products": {
        "description": "Catálogo de productos",
        "columns": [
            {"name": "product_id", "type": "string", "description": "SKU o ID único del producto"},
            {"name": "product_name", "type": "string", "description": "Nombre del producto"},
            {"name": "category", "type": "string", "description": "Categoría del producto"},
            {"name": "brand", "type": "string", "description": "Marca del producto"},
            {"name": "price", "type": "float", "description": "Precio en euros"},
            {"name": "cost", "type": "float", "description": "Costo de producción"},
            {"name": "weight_kg", "type": "float", "description": "Peso en kilogramos"},
            {"name": "dimensions", "type": "string", "description": "Dimensiones (LxWxH)"},
            {"name": "in_stock", "type": "integer", "description": "Cantidad en inventario"},
            {"name": "launch_date", "type": "date", "description": "Fecha de lanzamiento"}
        ]
    },
    "transactions": {
        "description": "Transacciones/Ventas",
        "columns": [
            {"name": "transaction_id", "type": "string", "description": "ID único de la transacción"},
            {"name": "customer_id", "type": "string", "description": "ID del cliente"},
            {"name": "product_id", "type": "string", "description": "ID del producto"},
            {"name": "quantity", "type": "integer", "description": "Cantidad comprada"},
            {"name": "unit_price", "type": "float", "description": "Precio unitario"},
            {"name": "total_amount", "type": "float", "description": "Monto total"},
            {"name": "discount_percent", "type": "float", "description": "Descuento aplicado (%)"},
            {"name": "payment_method", "type": "string", "description": "Método de pago"},
            {"name": "transaction_date", "type": "datetime", "description": "Fecha y hora de la transacción"},
            {"name": "status", "type": "string", "description": "Estado (completed, pending, cancelled)"}
        ]
    },
    "employees": {
        "description": "Datos de empleados",
        "columns": [
            {"name": "employee_id", "type": "string", "description": "ID único del empleado"},
            {"name": "first_name", "type": "string", "description": "Nombre del empleado"},
            {"name": "last_name", "type": "string", "description": "Apellido del empleado"},
            {"name": "email", "type": "string", "description": "Email corporativo"},
            {"name": "department", "type": "string", "description": "Departamento"},
            {"name": "position", "type": "string", "description": "Cargo/Posición"},
            {"name": "salary", "type": "float", "description": "Salario anual en euros"},
            {"name": "hire_date", "type": "date", "description": "Fecha de contratación"},
            {"name": "manager_id", "type": "string", "description": "ID del manager"},
            {"name": "office_location", "type": "string", "description": "Ubicación de la oficina"}
        ]
    },
    "insurance": {
        "description": "Pólizas de seguros",
        "columns": [
            {"name": "policy_id", "type": "string", "description": "ID único de la póliza"},
            {"name": "customer_id", "type": "string", "description": "ID del cliente"},
            {"name": "policy_type", "type": "string", "description": "Tipo de seguro (auto, hogar, vida, salud)"},
            {"name": "coverage_amount", "type": "float", "description": "Monto de cobertura"},
            {"name": "premium_annual", "type": "float", "description": "Prima anual"},
            {"name": "deductible", "type": "float", "description": "Deducible"},
            {"name": "start_date", "type": "date", "description": "Fecha de inicio"},
            {"name": "end_date", "type": "date", "description": "Fecha de fin"},
            {"name": "risk_score", "type": "integer", "description": "Puntuación de riesgo (1-10)"},
            {"name": "claims_count", "type": "integer", "description": "Número de reclamos"}
        ]
    }
}

# ==========================================
# HERRAMIENTAS PURE SYNTHETIC
# ==========================================

@function_tool
def list_data_templates() -> Dict[str, Any]:
    """
    Lista las plantillas de datos disponibles para generación desde cero.
    
    Returns:
        Diccionario con todas las plantillas disponibles
    """
    return {
        "success": True,
        "available_templates": DATA_TEMPLATES,
        "template_names": list(DATA_TEMPLATES.keys()),
        "usage_guide": {
            "customers": "Para e-commerce, CRM, marketing",
            "products": "Para inventarios, catálogos, e-commerce",
            "transactions": "Para análisis de ventas, finanzas",
            "employees": "Para HR, nómina, organización",
            "insurance": "Para aseguradoras, análisis de riesgo"
        }
    }

@function_tool
def get_template_details(template_name: str) -> Dict[str, Any]:
    """
    Obtiene los detalles de una plantilla específica.
    
    Args:
        template_name: Nombre de la plantilla
        
    Returns:
        Detalles completos de la plantilla
    """
    if template_name not in DATA_TEMPLATES:
        return {
            "success": False,
            "error": f"❌ Plantilla '{template_name}' no encontrada",
            "available_templates": list(DATA_TEMPLATES.keys())
        }
    
    template = DATA_TEMPLATES[template_name]
    return {
        "success": True,
        "template_name": template_name,
        "description": template["description"],
        "columns": template["columns"],
        "column_count": len(template["columns"]),
        "sample_structure": {col["name"]: f"<{col['type']}>" for col in template["columns"]}
    }

@function_tool
def create_custom_template(
    wrapper: RunContextWrapper[SyntheticDataContext],
    template_name: str,
    description: str,
    columns: List[Dict[str, str]]
) -> Dict[str, Any]:
    """
    Crea una plantilla personalizada de datos.
    
    Args:
        template_name: Nombre para la nueva plantilla
        description: Descripción de qué tipo de datos representa
        columns: Lista de columnas con name, type, description
        
    Returns:
        Confirmación de la plantilla creada
    """
    try:
        # Validar formato de columnas
        for col in columns:
            if not all(key in col for key in ["name", "type", "description"]):
                return {
                    "success": False,
                    "error": "❌ Cada columna debe tener: name, type, description"
                }
        
        # Crear plantilla personalizada
        custom_template = {
            "description": description,
            "columns": columns
        }
        
        # Guardar en contexto para uso posterior
        context = wrapper.context
        if not hasattr(context, 'custom_templates'):
            context.custom_templates = {}
        
        context.custom_templates[template_name] = custom_template
        context.add_to_history("custom_template_created", {
            "template_name": template_name,
            "column_count": len(columns)
        })
        
        return {
            "success": True,
            "template_name": template_name,
            "description": description,
            "columns": columns,
            "column_count": len(columns),
            "message": f"✅ Plantilla '{template_name}' creada exitosamente"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"❌ Error creando plantilla: {str(e)}"
        }

@function_tool
def generate_synthetic_data_from_scratch(
    wrapper: RunContextWrapper[SyntheticDataContext],
    template_name: str,
    num_rows: int,
    business_context: str = "",
    country: str = "Spain",
    model_name: str = "nemotron-mini"
) -> Dict[str, Any]:
    """
    Genera datos sintéticos desde cero usando LangChain + Nemotron.
    
    Args:
        template_name: Nombre de la plantilla a usar
        num_rows: Número de filas a generar
        business_context: Contexto del negocio para datos más realistas
        country: País para localización de datos
        model_name: Modelo de Ollama a usar
        
    Returns:
        Información del archivo generado
    """
    try:
        context = wrapper.context
        
        # Validaciones básicas
        if num_rows <= 0 or num_rows > 10000:
            return {
                "success": False,
                "error": "❌ Número de filas debe estar entre 1 y 10,000"
            }
        
        # Obtener plantilla (estándar o personalizada)
        template = None
        if template_name in DATA_TEMPLATES:
            template = DATA_TEMPLATES[template_name]
        elif hasattr(context, 'custom_templates') and template_name in context.custom_templates:
            template = context.custom_templates[template_name]
        else:
            return {
                "success": False,
                "error": f"❌ Plantilla '{template_name}' no encontrada",
                "available_templates": list(DATA_TEMPLATES.keys())
            }
        
        logger.info(f"🚀 Generando {num_rows} filas usando plantilla '{template_name}' con {model_name}")
        
        # Configurar Ollama LLM
        try:
            llm = get_ollama_llm(model_name)
        except Exception as e:
            return {
                "success": False,
                "error": f"❌ Error conectando con Ollama: {str(e)}"
            }
        
        # Crear prompt personalizado para el contexto
        columns_description = "\n".join([
            f"- {col['name']} ({col['type']}): {col['description']}"
            for col in template["columns"]
        ])
        
        business_prompt = f"""
        Context: {business_context if business_context else 'General business data'}
        Country/Region: {country}
        Industry: {template['description']}
        
        Generate realistic {template['description'].lower()} data with these columns:
        {columns_description}
        
        Make the data realistic for {country} market and ensure consistency across related fields.
        """
        
        # Configurar SyntheticDataGenerator de LangChain
        try:
            #generator = SyntheticDataGenerator(llm=llm)
            
            # Generar los datos usando el prompt personalizado
            synthetic_data = generator.generate(
                subject=template["description"],
                extra=business_prompt,
                runs=1  # Número de intentos de generación
            )
            
            # Procesar los datos generados
            if isinstance(synthetic_data, str):
                # Si es string, intentar parsear como JSON/CSV
                try:
                    # Intentar parsear como JSON
                    if synthetic_data.strip().startswith('[') or synthetic_data.strip().startswith('{'):
                        data_list = json.loads(synthetic_data)
                        df = pd.DataFrame(data_list)
                    else:
                        # Intentar parsear como CSV
                        from io import StringIO
                        df = pd.read_csv(StringIO(synthetic_data))
                except:
                    return {
                        "success": False,
                        "error": "❌ Error procesando datos generados por LLM",
                        "raw_output": synthetic_data[:500] + "..." if len(synthetic_data) > 500 else synthetic_data
                    }
            elif isinstance(synthetic_data, list):
                df = pd.DataFrame(synthetic_data)
            else:
                df = synthetic_data
            
            # Ajustar al número de filas solicitado
            if len(df) > num_rows:
                df = df.head(num_rows)
            elif len(df) < num_rows:
                # Si necesitamos más filas, replicar y variar los datos
                while len(df) < num_rows:
                    additional_rows = min(len(df), num_rows - len(df))
                    df_extra = df.head(additional_rows).copy()
                    
                    # Agregar variación aleatoria a campos numéricos
                    for col in df_extra.columns:
                        if df_extra[col].dtype in ['int64', 'float64']:
                            noise = np.random.normal(0, 0.1, len(df_extra))
                            df_extra[col] = df_extra[col] * (1 + noise)
                    
                    df = pd.concat([df, df_extra], ignore_index=True)
            
            df = df.head(num_rows)  # Asegurar número exacto
            
        except Exception as e:
            logger.error(f"Error generando datos con LangChain: {e}")
            return {
                "success": False,
                "error": f"❌ Error generando datos: {str(e)}"
            }
        
        # Crear archivo con nombre descriptivo
        import pandas as pd
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{template_name}_synthetic_{num_rows}rows_{timestamp}.csv"
        
        # Guardar en directorio de trabajo actual
        current_dir = os.getcwd()
        output_path = os.path.join(current_dir, output_filename)
        
        # Guardar archivo CSV
        df.to_csv(output_path, index=False)
        file_size_mb = round(os.path.getsize(output_path) / 1024 / 1024, 2)
        
        # Actualizar contexto
        context.generated_file_path = output_path
        context.generated_file_id = timestamp
        context.generated_rows = len(df)
        context.last_model_used = f"LangChain+{model_name}"
        context.add_to_history("pure_synthetic_generated", {
            "template_name": template_name,
            "num_rows": len(df),
            "model_used": model_name,
            "file_size_mb": file_size_mb
        })
        
        result = {
            "success": True,
            "file_id": timestamp,
            "output_filename": output_filename,
            "output_path": output_path,
            "rows_generated": len(df),
            "columns": list(df.columns),
            "template_used": template_name,
            "model_used": f"LangChain + {model_name}",
            "business_context": business_context,
            "country": country,
            "file_size_mb": file_size_mb,
            "sample_data": df.head(3).to_dict('records'),
            "generation_summary": f"✅ Generadas {len(df):,} filas usando plantilla '{template_name}' con {model_name}",
            "access_instructions": f"Archivo guardado en: {output_path}"
        }
        
        logger.info(f"✅ Datos sintéticos generados desde cero: {output_filename} ({file_size_mb}MB)")
        return result
        
    except Exception as e:
        error_msg = f"❌ Error generando datos desde cero: {str(e)}"
        logger.error(error_msg)
        return {"success": False, "error": error_msg}

@function_tool
def test_ollama_connection(model_name: str = "nemotron-mini") -> Dict[str, Any]:
    """
    Prueba la conexión con Ollama y el modelo especificado.
    
    Args:
        model_name: Nombre del modelo a probar
        
    Returns:
        Estado de la conexión y modelo
    """
    try:
        llm = get_ollama_llm(model_name)
        
        # Test simple
        test_prompt = "Generate a simple greeting."
        response = llm.invoke(test_prompt)
        
        return {
            "success": True,
            "model_name": model_name,
            "status": "Connected",
            "test_response": response[:100] + "..." if len(response) > 100 else response,
            "message": f"✅ Conexión exitosa con {model_name}"
        }
        
    except Exception as e:
        return {
            "success": False,
            "model_name": model_name,
            "status": "Failed",
            "error": str(e),
            "message": f"❌ Error conectando con {model_name}: {str(e)}"
        }