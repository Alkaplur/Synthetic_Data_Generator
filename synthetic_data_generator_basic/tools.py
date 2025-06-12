"""
Tools for Synthetic Data Generator - Basic Version
Herramientas específicas para generación de datos sintéticos
"""

import os
import logging
import pandas as pd
import uuid
from typing import Dict, Any, Optional, Callable
import inspect
import asyncio

from context import SyntheticDataContext

# Setup logging
logger = logging.getLogger(__name__)


class Tool:
    """Representa una herramienta que puede ser llamada por un agente"""
    
    def __init__(self, func: Callable, name: str = None):
        self.func = func
        self.name = name or func.__name__.replace("_tool", "")
        self.description = func.__doc__ or ""
        self.signature = inspect.signature(func)
    
    async def call(self, context: SyntheticDataContext, **kwargs) -> Dict[str, Any]:
        """Ejecuta la herramienta con contexto y argumentos"""
        try:
            # Bind arguments
            sig = self.signature
            bound_args = sig.bind_partial(context, **kwargs)
            bound_args.apply_defaults()
            
            # Call function
            if asyncio.iscoroutinefunction(self.func):
                result = await self.func(*bound_args.args, **bound_args.kwargs)
            else:
                result = self.func(*bound_args.args, **bound_args.kwargs)
            
            return {"success": True, "result": result}
            
        except Exception as e:
            logger.error(f"Error calling tool {self.name}: {str(e)}")
            return {"success": False, "error": str(e)}


def function_tool(func: Callable) -> Tool:
    """Decorador para convertir una función en Tool"""
    return Tool(func)


# ==========================================
# HERRAMIENTAS ESPECÍFICAS DEL DOMINIO
# ==========================================

@function_tool
async def load_and_analyze_data_tool(
    context: SyntheticDataContext,
    file_path: str
) -> Dict[str, Any]:
    """
    Carga y analiza un archivo CSV.
    
    Args:
        context: El contexto de la conversación
        file_path: Ruta al archivo CSV
        
    Returns:
        Dict con resultados del análisis
    """
    try:
        df = pd.read_csv(file_path)
        context.source_data = df
        context.source_file_path = file_path
        
        return {
            "rows": len(df),
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.to_dict().items()},
            "missing_values": df.isnull().sum().to_dict(),
            "memory_usage": df.memory_usage(deep=True).sum()
        }
    except Exception as e:
        logger.error(f"Error loading file: {str(e)}")
        raise


@function_tool
async def sdv_generate_tool(
    context: SyntheticDataContext,
    num_rows: int
) -> Dict[str, Any]:
    """
    Genera datos sintéticos usando SDV.
    
    Args:
        context: El contexto de la conversación
        num_rows: Número de filas a generar
        
    Returns:
        Dict con resultados de la generación
    """
    try:
        if context.source_data is None:
            raise ValueError("No hay datos fuente disponibles. Carga un archivo primero.")
        
        # Import SDV (lazy loading)
        try:
            from sdv.tabular import GaussianCopula
        except ImportError:
            raise ImportError("SDV no está instalado. Instala con: pip install sdv")
        
        # Train model
        model = GaussianCopula()
        model.fit(context.source_data)
        
        # Generate data
        synthetic_data = model.sample(num_rows)
        
        # Save to file
        file_id = str(uuid.uuid4())
        file_path = os.path.join(context.temp_dir, f"synthetic_{file_id}.csv")
        synthetic_data.to_csv(file_path, index=False)
        
        # Update context
        context.generated_file_id = file_id
        context.generated_file_path = file_path
        context.generated_rows = num_rows
        
        return {
            "file_id": file_id,
            "rows": num_rows,
            "file_path": file_path,
            "columns": list(synthetic_data.columns)
        }
        
    except Exception as e:
        logger.error(f"Error generating data with SDV: {str(e)}")
        raise


@function_tool
async def llm_generate_tool(
    context: SyntheticDataContext,
    num_rows: int,
    data_schema: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Genera datos sintéticos usando LLM (placeholder).
    
    Args:
        context: El contexto de la conversación
        num_rows: Número de filas a generar
        data_schema: Esquema de datos opcional
        
    Returns:
        Dict con resultados de la generación
    """
    try:
        # TODO: Implementar generación real con LLM
        # Por ahora, generar datos de ejemplo
        
        import random
        import datetime
        
        # Schema por defecto si no se proporciona
        if not data_schema:
            data_schema = {
                "customer_id": "string",
                "name": "string", 
                "age": "integer",
                "email": "email",
                "purchase_amount": "float",
                "registration_date": "date"
            }
        
        # Generar datos de ejemplo
        synthetic_rows = []
        for i in range(num_rows):
            row = {}
            for col, dtype in data_schema.items():
                if dtype == "string":
                    row[col] = f"Customer_{i+1}"
                elif dtype == "email":
                    row[col] = f"user{i+1}@example.com"
                elif dtype == "integer":
                    row[col] = random.randint(18, 80)
                elif dtype == "float":
                    row[col] = round(random.uniform(10.0, 1000.0), 2)
                elif dtype == "date":
                    start_date = datetime.date(2020, 1, 1)
                    random_days = random.randint(0, 1000)
                    row[col] = start_date + datetime.timedelta(days=random_days)
            synthetic_rows.append(row)
        
        # Crear DataFrame
        synthetic_data = pd.DataFrame(synthetic_rows)
        
        # Save to file
        file_id = str(uuid.uuid4())
        file_path = os.path.join(context.temp_dir, f"llm_synthetic_{file_id}.csv")
        synthetic_data.to_csv(file_path, index=False)
        
        # Update context
        context.generated_file_id = file_id
        context.generated_file_path = file_path
        context.generated_rows = num_rows
        
        return {
            "file_id": file_id,
            "rows": num_rows,
            "file_path": file_path,
            "columns": list(synthetic_data.columns),
            "method": "LLM_placeholder"
        }
        
    except Exception as e:
        logger.error(f"Error generating data with LLM: {str(e)}")
        raise


@function_tool
async def create_download_link_tool(
    context: SyntheticDataContext
) -> Dict[str, Any]:
    """
    Crea un enlace de descarga para datos generados.
    
    Args:
        context: El contexto de la conversación
        
    Returns:
        Dict con información de descarga
    """
    try:
        if not context.generated_file_id:
            raise ValueError("No se ha generado ningún archivo aún")
            
        return {
            "file_id": context.generated_file_id,
            "rows": context.generated_rows,
            "download_url": f"/download/{context.generated_file_id}?session_id={context.session_id}",
            "file_exists": os.path.exists(context.generated_file_path) if context.generated_file_path else False
        }
        
    except Exception as e:
        logger.error(f"Error creating download link: {str(e)}")
        raise


@function_tool
async def get_file_info_tool(
    context: SyntheticDataContext
) -> Dict[str, Any]:
    """
    Obtiene información sobre archivos generados.
    
    Args:
        context: El contexto de la conversación
        
    Returns:
        Dict con información de archivos
    """
    try:
        return {
            "has_source_file": bool(context.source_file_path),
            "source_file_path": context.source_file_path,
            "has_generated_file": bool(context.generated_file_id),
            "generated_file_id": context.generated_file_id,
            "generated_rows": context.generated_rows,
            "temp_dir": context.temp_dir,
            "session_id": context.session_id
        }
        
    except Exception as e:
        logger.error(f"Error getting file info: {str(e)}")
        raise


# ==========================================
# UTILIDADES PARA AGENTES
# ==========================================

def get_available_tools() -> Dict[str, Tool]:
    """Retorna todas las herramientas disponibles"""
    return {
        "load_and_analyze": load_and_analyze_data_tool,
        "sdv_generate": sdv_generate_tool,
        "llm_generate": llm_generate_tool,
        "create_download": create_download_link_tool,
        "get_file_info": get_file_info_tool
    }


def get_tools_for_agent(agent_type: str) -> list[Tool]:
    """Retorna herramientas específicas para un tipo de agente"""
    all_tools = get_available_tools()
    
    if agent_type == "sample_data":
        return [
            all_tools["load_and_analyze"],
            all_tools["sdv_generate"],
            all_tools["create_download"],
            all_tools["get_file_info"]
        ]
    elif agent_type == "pure_synthetic":
        return [
            all_tools["llm_generate"],
            all_tools["create_download"],
            all_tools["get_file_info"]
        ]
    elif agent_type == "orchestrator":
        return [all_tools["get_file_info"]]
    else:
        return list(all_tools.values())