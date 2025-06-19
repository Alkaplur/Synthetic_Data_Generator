"""
Tools for Synthetic Data Generator - With SDK Context Management
Herramientas que usan el contexto del SDK para mantener estado entre llamadas
"""

import os
import logging
import pandas as pd
from pathlib import Path
import tempfile
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from agents import function_tool, RunContextWrapper

# Setup logging
logger = logging.getLogger(__name__)

# ==========================================
# CONTEXTO PARA EL SDK - DEFINICIÓN ÚNICA
# ==========================================
# Directorio base del proyecto  
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "Synthetic data generated"

@dataclass
class SyntheticDataContext:
    """
    Contexto que se pasa al SDK y mantiene estado entre herramientas
    """
    user_id: str
    session_id: str
    temp_dir: str = field(default_factory=lambda: tempfile.mkdtemp(prefix="synthetic_data_"))

    # 🧠 MEMORIA DE CONVERSACIÓN 
    conversation_messages: list = field(default_factory=list)
    
    # Estado de archivos
    analyzed_file_path: Optional[str] = None
    analyzed_file_info: Optional[Dict[str, Any]] = None
    generated_file_path: Optional[str] = None
    generated_file_id: Optional[str] = None
    generated_rows: Optional[int] = None
    
    # Metadatos de proceso
    last_model_used: Optional[str] = None
    processing_history: list = field(default_factory=list)
    current_agent: Optional[str] = None
    
    def add_to_history(self, action: str, details: Dict[str, Any]):
        """Agregar acción al historial"""
        self.processing_history.append({
            "timestamp": pd.Timestamp.now().isoformat(),
            "action": action,
            "details": details
        })

# ==========================================
# HERRAMIENTAS CON CONTEXTO DEL SDK
# ==========================================

@function_tool
def analyze_csv_file(wrapper: RunContextWrapper[SyntheticDataContext], file_path: str) -> Dict[str, Any]:
    """
    Analiza un archivo CSV y guarda la información en el contexto para uso posterior.
    
    Args:
        file_path: Ruta completa al archivo CSV a analizar
        
    Returns:
        Diccionario con estadísticas del archivo CSV
    """
    try:
        # Obtener contexto del SDK
        context = wrapper.context
        
        # Cargar CSV
        df = pd.read_csv(file_path)
        
        # Análisis completo del archivo
        analysis = {
            "success": True,
            "file_path": file_path,
            "filename": os.path.basename(file_path),
            "rows": len(df),
            "columns": list(df.columns),
            "column_count": len(df.columns),
            "data_types": {col: str(dtype) for col, dtype in df.dtypes.to_dict().items()},
            "missing_values": df.isnull().sum().to_dict(),
            "missing_percentage": (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
            "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
            "numeric_columns": df.select_dtypes(include=['number']).columns.tolist(),
            "categorical_columns": df.select_dtypes(include=['object']).columns.tolist(),
            "sample_data": df.head(3).to_dict('records'),
            "basic_stats": df.describe().to_dict() if len(df.select_dtypes(include=['number']).columns) > 0 else {}
        }
        
        # 🎯 GUARDAR EN CONTEXTO para uso posterior
        context.analyzed_file_path = file_path
        context.analyzed_file_info = analysis
        context.add_to_history("csv_analyzed", {
            "file_path": file_path,
            "rows": len(df),
            "columns": len(df.columns)
        })
        
        logger.info(f"✅ CSV analizado y guardado en contexto: {os.path.basename(file_path)} - {len(df)} filas, {len(df.columns)} columnas")
        return analysis
        
    except FileNotFoundError:
        error_msg = f"❌ Archivo no encontrado: {file_path}"
        logger.error(error_msg)
        return {"success": False, "error": error_msg}
    except pd.errors.EmptyDataError:
        error_msg = "❌ El archivo CSV está vacío"
        logger.error(error_msg)
        return {"success": False, "error": error_msg}
    except pd.errors.ParserError as e:
        error_msg = f"❌ Error parsing CSV: {str(e)}"
        logger.error(error_msg)
        return {"success": False, "error": error_msg}
    except Exception as e:
        error_msg = f"❌ Error inesperado analizando CSV: {str(e)}"
        logger.error(error_msg)
        return {"success": False, "error": error_msg}


@function_tool
def generate_synthetic_data_with_sdv(
    wrapper: RunContextWrapper[SyntheticDataContext],
    num_rows: int,
    model_type: str = "GaussianCopula"
) -> Dict[str, Any]:
    """
    Genera datos sintéticos usando SDV. Usa automáticamente el archivo previamente analizado.
    
    Args:
        num_rows: Número de filas sintéticas a generar
        model_type: Tipo de modelo SDV (GaussianCopula, CTGAN, CopulaGAN, TVAE)
        
    Returns:
        Diccionario con información del archivo generado
    """
    try:
        # Obtener contexto del SDK
        context = wrapper.context
        
        # 🎯 USAR ARCHIVO DEL CONTEXTO
        if not context.analyzed_file_path:
            return {
                "success": False,
                "error": "❌ Primero debes analizar un archivo CSV usando analyze_csv_file()"
            }
        
        source_file_path = context.analyzed_file_path
        
        # Cargar datos fuente
        source_df = pd.read_csv(source_file_path)
        
        # Validaciones básicas
        if len(source_df) < 2:
            return {
                "success": False,
                "error": "❌ Se necesitan al menos 2 filas en los datos fuente para entrenar el modelo"
            }
        
        if num_rows <= 0:
            return {
                "success": False,
                "error": "❌ El número de filas debe ser mayor a 0"
            }
        
        if num_rows > 100000:
            return {
                "success": False,
                "error": "❌ Máximo 100,000 filas por generación para evitar problemas de memoria"
            }
        
        # Import SDV basado en el modelo seleccionado (SDV 1.21+)
        try:
            if model_type == "GaussianCopula":
                from sdv.single_table import GaussianCopulaSynthesizer as Synthesizer
            elif model_type == "CTGAN":
                from sdv.single_table import CTGANSynthesizer as Synthesizer
            elif model_type == "CopulaGAN":
                from sdv.single_table import CopulaGANSynthesizer as Synthesizer
            elif model_type == "TVAE":
                from sdv.single_table import TVAESynthesizer as Synthesizer
            else:
                return {
                    "success": False,
                    "error": f"❌ Modelo no soportado: {model_type}. Modelos disponibles: GaussianCopula, CTGAN, CopulaGAN, TVAE"
                }
        except ImportError as e:
            return {
                "success": False,
                "error": "❌ SDV no está instalado correctamente. Instala con: pip install sdv"
            }

        logger.info(f"🚀 Iniciando entrenamiento {model_type} con {len(source_df)} filas fuente")

        # Crear metadatos para SDV 1.21+ (algunos modelos los requieren)
        try:
            from sdv.metadata import SingleTableMetadata
            metadata = SingleTableMetadata()
            metadata.detect_from_dataframe(source_df)
            logger.info(f"📊 Metadatos detectados: {len(metadata.columns)} columnas")
        except Exception as e:
            logger.error(f"Error creando metadatos: {str(e)}")
            return {
                "success": False,
                "error": f"❌ Error creando metadatos: {str(e)}"
            }

        # Crear synthesizer con manejo específico por modelo
        try:
            if model_type == "GaussianCopula":
                # GaussianCopula puede funcionar sin metadatos
                try:
                    synthesizer = Synthesizer()
                except:
                    synthesizer = Synthesizer(metadata)
            else:
                # CTGAN, CopulaGAN, TVAE requieren metadatos
                synthesizer = Synthesizer(metadata)
            
            logger.info(f"✅ Synthesizer {model_type} creado exitosamente")
            
            # Entrenar modelo
            synthesizer.fit(source_df)
            logger.info(f"✅ Modelo {model_type} entrenado exitosamente")
            
        except Exception as e:
            logger.error(f"Error entrenando modelo {model_type}: {str(e)}")
            return {
                "success": False,
                "error": f"❌ Error entrenando modelo {model_type}: {str(e)}"
            }
        
        logger.info(f"🎯 Generando {num_rows} filas sintéticas...")
        
        # Generar datos sintéticos
        try:
            synthetic_data = synthesizer.sample(num_rows)
            logger.info(f"✅ {num_rows} filas sintéticas generadas exitosamente")
        except Exception as e:
            logger.error(f"Error generando datos: {str(e)}")
            return {
                "success": False,
                "error": f"❌ Error generando {num_rows} filas: {str(e)}"
            }
        
        # Crear archivo con nombre descriptivo en directorio de trabajo
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        source_filename = os.path.splitext(os.path.basename(source_file_path))[0]
        output_filename = f"{source_filename}_synthetic_data_{model_type.lower()}_{num_rows}rows_{timestamp}.csv"
        
        # Guardar en directorio específico para datos sintéticos
        output_dir = "/Users/davidnogueras/Desktop/Cursor/Synthetic_Data_Generator/synthetic_data_generator_basic/Synthetic data generated"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_filename)
        
        # Guardar archivo CSV
        synthetic_data.to_csv(output_path, index=False)
        
        # Calcular métricas básicas
        file_size_mb = round(os.path.getsize(output_path) / 1024 / 1024, 2)
        
        # 🎯 ACTUALIZAR CONTEXTO
        context.generated_file_path = output_path
        context.generated_file_id = timestamp
        context.generated_rows = num_rows
        context.last_model_used = model_type
        context.add_to_history("synthetic_data_generated", {
            "model_type": model_type,
            "num_rows": num_rows,
            "output_filename": output_filename,
            "file_size_mb": file_size_mb,
            "saved_to": "synthetic_data_generated_directory"
        })
        
        result = {
            "success": True,
            "file_id": timestamp,
            "output_filename": output_filename,
            "output_path": output_path,
            "saved_in_specific_directory": True,
            "full_path_for_access": output_path,
            "rows_generated": num_rows,
            "columns": list(synthetic_data.columns),
            "model_used": model_type,
            "source_file": os.path.basename(source_file_path),
            "source_rows": len(source_df),
            "file_size_mb": file_size_mb,
            "sample_synthetic_data": synthetic_data.head(3).to_dict('records'),
            "generation_summary": f"✅ Generadas {num_rows:,} filas sintéticas usando {model_type} desde {len(source_df):,} filas originales",
            "access_instructions": f"El archivo se guardó en: {output_path}"
        }
        
        logger.info(f"✅ Datos sintéticos generados y guardados en contexto: {output_filename} ({file_size_mb}MB)")
        return result
        
    except Exception as e:
        error_msg = f"❌ Error generando datos con SDV: {str(e)}"
        logger.error(error_msg)
        return {"success": False, "error": error_msg}
    

@function_tool
def get_session_status(wrapper: RunContextWrapper[SyntheticDataContext]) -> Dict[str, Any]:
    """
    Obtiene el estado completo de la sesión actual.
    
    Returns:
        Estado completo de la sesión y archivos
    """
    try:
        # Obtener contexto del SDK
        context = wrapper.context
        
        session_status = {
            "success": True,
            "session_info": {
                "session_id": context.session_id,
                "user_id": context.user_id,
                "temp_directory": context.temp_dir
            },
            "analyzed_file": {
                "has_file": bool(context.analyzed_file_path),
                "file_path": context.analyzed_file_path,
                "file_info": context.analyzed_file_info
            },
            "generated_file": {
                "has_file": bool(context.generated_file_id),
                "file_id": context.generated_file_id,
                "file_path": context.generated_file_path,
                "rows_generated": context.generated_rows,
                "model_used": context.last_model_used
            },
            "processing_history": context.processing_history[-5:] if context.processing_history else [],  # Últimas 5 acciones
            "total_actions": len(context.processing_history)
        }
        
        return session_status
        
    except Exception as e:
        error_msg = f"❌ Error obteniendo estado de sesión: {str(e)}"
        logger.error(error_msg)
        return {"success": False, "error": error_msg}


# ==========================================
# UTILIDADES PARA AGENTES - CORREGIDAS
# ==========================================

def get_tools_for_agent(agent_type: str) -> list:
    """
    Retorna herramientas específicas para un tipo de agente.
    
    Args:
        agent_type: Tipo de agente
        
    Returns:
        Lista de function_tools para el agente
    """
    if agent_type == "sample_data":
        return [
            analyze_csv_file,
            generate_synthetic_data_with_sdv,
            get_session_status
        ]
    elif agent_type == "pure_synthetic":
        # Import local para evitar circular import
        from nvidia_synthetic_generator import generate_synthetic_data_dynamic
        return [
            generate_synthetic_data_dynamic,
            get_session_status
        ]
    elif agent_type == "orchestrator":
        return [
            get_session_status
        ]
    elif agent_type == "pure_historical":
        return [
            # TODO: Agregar herramientas para carga de datos históricos
            get_session_status
        ]
    else:
        # Retornar todas las herramientas disponibles
        return [
            analyze_csv_file,
            list_sdv_models,
            generate_synthetic_data_with_sdv,
            create_download_link,
            get_session_status
        ]