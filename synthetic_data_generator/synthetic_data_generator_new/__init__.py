# synthetic_data_generator/__init__.py
"""
Synthetic Data Generator - Generador de Datos Sintéticos Dinámico
================================================================

Una herramienta flexible para generar datos sintéticos de cualquier estructura
utilizando SDV y LangChain con un sistema de agentes coordinados.

Características principales:
- 🎯 Generación dinámica desde descripciones en lenguaje natural
- 📊 Generación adaptativa desde cualquier muestra de datos
- 🔍 Análisis automático de patrones y estructura
- 📈 Visualización automática de datos
- 💾 Exportación inteligente en múltiples formatos
- 🤖 Sistema de agentes que se adapta al contexto

Filosofía de diseño:
- NO estructuras predefinidas
- Adaptación automática a cualquier tipo de datos
- Generación basada en contexto y patrones detectados
- API simple pero potente

Uso básico - Generación dinámica:
    ```python
    from synthetic_data_generator import generate_data
    
    # Generación completamente libre desde descripción
    result = generate_data(
        "Generar datos de ventas de una tienda online con productos, 
         clientes, fechas, cantidades y cualquier campo relevante"
    )
    
    # Generación adaptativa desde cualquier muestra
    sample = {
        "transacciones": [
            {"producto": "Laptop", "precio": 1200, "cliente": "Juan", "fecha": "2024-01-15"},
            {"producto": "Mouse", "precio": 25, "cliente": "Ana", "fecha": "2024-01-16"}
        ]
    }
    result = generate_data(
        "Generar más transacciones similares pero con variedad", 
        sample_data=sample,
        count=1000
    )
    ```
"""

import warnings
import sys
from typing import Dict, Any, Optional, Union, List
import pandas as pd
import asyncio

# Información del paquete
__version__ = "1.0.0"
__author__ = "Tu Nombre"
__email__ = "tu.email@ejemplo.com"
__description__ = "Generador dinámico de datos sintéticos con SDV y LangChain"
__url__ = "https://github.com/Alkaplur/Synthetic_Data_Generator"

# Imports principales con manejo de errores durante desarrollo
try:
    from .core.base_generator import BaseGenerator
    from .core.sdv_generator import SDVGenerator
    from .core.llm_generator import LLMGenerator
    from .agents.coordinator import AgentCoordinator
    from .config.settings import get_settings, update_settings
    from .utils.exceptions import SyntheticDataError, GenerationError, ConfigurationError
    from .utils.data_detector import DataStructureDetector
    from .utils.formatters import DynamicFormatter
except ImportError as e:
    warnings.warn(f"Algunos módulos no están disponibles: {e}")
    
    # Clases placeholder para desarrollo
    class BaseGenerator:
        def __init__(self, config=None): pass
    class SDVGenerator(BaseGenerator): pass
    class LLMGenerator(BaseGenerator): pass
    class AgentCoordinator:
        def __init__(self): pass
        async def process_request(self, **kwargs): 
            return {"data": pd.DataFrame(), "success": False}
    
    class SyntheticDataError(Exception): pass
    class GenerationError(SyntheticDataError): pass
    class ConfigurationError(SyntheticDataError): pass
    
    class DataStructureDetector:
        def detect_structure(self, data): return {}
    class DynamicFormatter:
        def auto_format(self, data, hint=None): return data
    
    def get_settings(): return {}
    def update_settings(**kwargs): pass

# API pública
__all__ = [
    'SyntheticDataGenerator',
    'generate_data',
    'configure',
    'SyntheticDataError',
    'GenerationError', 
    'ConfigurationError',
    '__version__'
]

class SyntheticDataGenerator:
    """
    Generador dinámico de datos sintéticos
    
    Esta clase NO tiene estructuras predefinidas. Se adapta automáticamente
    a cualquier tipo de datos o descripción que se le proporcione.
    
    Ejemplos de uso dinámico:
    
        # Generación libre - El sistema decide la estructura
        generator = SyntheticDataGenerator()
        
        # Desde descripción natural - estructura dinámica
        data = generator.generate(
            "Crear datos de una empresa de logística con envíos, 
             conductores, rutas, tiempos de entrega, costos y cualquier 
             otro campo que consideres relevante para el análisis"
        )
        
        # Desde muestra - adaptación automática
        sample = pd.DataFrame({
            'sensor_id': ['A001', 'B002'],
            'temperature': [23.5, 25.1],
            'humidity': [45.2, 48.7],
            'timestamp': ['2024-01-01 10:00', '2024-01-01 10:05'],
            'location': ['Warehouse_A', 'Warehouse_B']
        })
        
        more_data = generator.generate(
            "Generar más datos de sensores pero incluyendo nuevos tipos 
             de mediciones y ubicaciones creativas",
            sample_data=sample,
            count=5000
        )
        
        # Generación híbrida - combina descripción + muestra
        hybrid_data = generator.generate(
            "Expandir estos datos agregando análisis predictivo, 
             alertas automáticas y cualquier campo que ayude en IoT",
            sample_data=sample,
            expand_structure=True
        )
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Inicializar generador dinámico"""
        self.config = config
        self._coordinator = None
        self._structure_detector = DataStructureDetector()
        self._formatter = DynamicFormatter()
        
        # Verificar configuración automáticamente
        self._validate_config()
    
    def _validate_config(self):
        """Validar configuración desde config.py"""
        try:
            settings = get_settings()
            if not settings.get('openai_api_key') and not settings.get('llm', {}).get('api_key'):
                warnings.warn(
                    "⚠️  API Key no encontrada. Configure en config/settings.py o use configure()\n"
                    "   La generación LLM no funcionará sin API key."
                )
        except Exception:
            pass
    
    @property
    def coordinator(self) -> AgentCoordinator:
        """Lazy loading del coordinador con configuración automática"""
        if self._coordinator is None:
            settings = get_settings()
            self._coordinator = AgentCoordinator(config=settings)
        return self._coordinator
    
    def generate(self, 
                description_or_request: str,
                sample_data: Optional[Union[pd.DataFrame, Dict, List[Dict], Any]] = None,
                count: int = 100,
                expand_structure: bool = False,
                creativity_level: float = 0.7,
                **kwargs) -> pd.DataFrame:
        """
        Generación completamente dinámica de datos sintéticos
        
        Args:
            description_or_request: Descripción libre de lo que necesitas.
                                  Puede ser tan específico o general como quieras.
            sample_data: Cualquier tipo de datos de muestra (opcional).
                        El sistema se adaptará automáticamente.
            count: Número de registros a generar
            expand_structure: Si True, expande la estructura más allá de la muestra
            creativity_level: Nivel de creatividad (0.0-1.0)
            **kwargs: Parámetros adicionales que se pasan al generador
            
        Returns:
            DataFrame con estructura dinámica adaptada al contexto
            
        El sistema automáticamente:
        - Detecta el tipo de datos y estructura
        - Elige el mejor método de generación
        - Adapta la salida al contexto
        - Valida la coherencia de los datos generados
        """
        try:
            # Preparar el contexto dinámico
            context = self._build_dynamic_context(
                description_or_request, 
                sample_data, 
                expand_structure,
                creativity_level,
                **kwargs
            )
            
            # Usar el coordinador para procesamiento inteligente
            result = asyncio.run(
                self.coordinator.process_request(
                    request=description_or_request,
                    sample_data=sample_data,
                    count=count,
                    context=context,
                    **kwargs
                )
            )
            
            if result.get('success'):
                return result['data']
            else:
                raise GenerationError(f"Error en generación: {result.get('error')}")
                
        except Exception as e:
            raise GenerationError(f"Error en generación dinámica: {str(e)}")
    
    def _build_dynamic_context(self, 
                              description: str,
                              sample_data: Any,
                              expand_structure: bool,
                              creativity_level: float,
                              **kwargs) -> Dict[str, Any]:
        """Construir contexto dinámico para la generación"""
        
        context = {
            'description': description,
            'expand_structure': expand_structure,
            'creativity_level': creativity_level,
            'generation_hints': [],
            'detected_patterns': {},
            'suggested_fields': [],
            'data_type': 'unknown'
        }
        
        # Detectar estructura y patrones si hay muestra
        if sample_data is not None:
            try:
                structure_info = self._structure_detector.detect_structure(sample_data)
                context.update({
                    'detected_patterns': structure_info.get('patterns', {}),
                    'data_type': structure_info.get('type', 'tabular'),
                    'suggested_fields': structure_info.get('suggested_fields', []),
                    'sample_size': structure_info.get('sample_size', 0)
                })
            except Exception as e:
                warnings.warn(f"No se pudo analizar la estructura de la muestra: {e}")
        
        # Añadir hints basados en la descripción
        context['generation_hints'] = self._extract_hints_from_description(description)
        
        return context
    
    async def _extract_hints_from_description(self, description: str) -> List[str]:
        """
        Extraer hints de generación usando LLM - Análisis inteligente y flexible
        
        En lugar de reglas rígidas, usa IA para entender el contexto real
        """
        try:
            # Prompt especializado para análisis de contexto
            analysis_prompt = f"""
            Analiza esta solicitud de generación de datos sintéticos y extrae información clave:

            SOLICITUD: "{description}"

            Responde SOLO con un JSON con esta estructura:
            {{
                "domain": "el dominio principal (ej: healthcare, finance, ecommerce, iot, etc)",
                "data_type": "tipo de datos (transactional, user_profiles, time_series, etc)",  
                "intent": "intención (replicate, expand, create_new, enhance, etc)",
                "complexity": "simple|medium|complex",
                "suggested_fields": ["campo1", "campo2", "campo3"],
                "context_hints": ["hint1", "hint2"],
                "creativity_level": 0.7
            }}

            Ejemplos:
            - "empleados de empresa" → domain: "human_resources", suggested_fields: ["name", "age", "department", "salary"]
            - "sensores IoT temperatura" → domain: "iot", data_type: "time_series", suggested_fields: ["sensor_id", "temperature", "timestamp", "location"]
            - "más datos similares" → intent: "replicate"
            - "expandir con nuevos campos" → intent: "expand"
            """
            
            # Usar LLM para análisis (implementación depende del generador LLM disponible)
            try:
                # Intentar usar el generador LLM existente
                if hasattr(self, '_llm_analyzer') and self._llm_analyzer:
                    analysis_result = await self._llm_analyzer.quick_analyze(analysis_prompt)
                else:
                    # Usar coordinador para análisis rápido
                    analysis_result = await self.coordinator.quick_analysis(analysis_prompt)
                
                # Parsear respuesta JSON
                import json
                if isinstance(analysis_result, str):
                    analysis_data = json.loads(analysis_result)
                else:
                    analysis_data = analysis_result
                
                # Convertir a lista de hints formato consistente
                hints = []
                if analysis_data.get('domain'):
                    hints.append(f"domain:{analysis_data['domain']}")
                if analysis_data.get('data_type'):
                    hints.append(f"data_type:{analysis_data['data_type']}")
                if analysis_data.get('intent'):
                    hints.append(f"intent:{analysis_data['intent']}")
                if analysis_data.get('complexity'):
                    hints.append(f"complexity:{analysis_data['complexity']}")
                
                # Agregar campos sugeridos como hints
                for field in analysis_data.get('suggested_fields', []):
                    hints.append(f"suggested_field:{field}")
                
                # Agregar hints de contexto
                for hint in analysis_data.get('context_hints', []):
                    hints.append(f"context:{hint}")
                
                return hints
                
            except Exception as llm_error:
                warnings.warn(f"Error en análisis LLM: {llm_error}, usando análisis básico")
                return self._fallback_simple_analysis(description)
                
        except Exception as e:
            warnings.warn(f"Error en extracción de hints: {e}")
            return [f"raw_description:{description}"]
    
    def _fallback_simple_analysis(self, description: str) -> List[str]:
        """
        Análisis de respaldo simple cuando LLM no está disponible
        Mucho más simple que la versión anterior
        """
        hints = [f"raw_description:{description}"]
        
        # Solo algunos patrones muy obvios como fallback
        desc_lower = description.lower()
        
        if any(word in desc_lower for word in ['empleado', 'employee', 'worker', 'staff']):
            hints.append('domain:human_resources')
        elif any(word in desc_lower for word in ['venta', 'sale', 'customer', 'producto']):
            hints.append('domain:sales')
        elif any(word in desc_lower for word in ['sensor', 'iot', 'temperatura', 'medición']):
            hints.append('domain:iot')
        
        return hints
    
    def analyze_sample(self, sample_data: Union[pd.DataFrame, Dict, List, Any]) -> Dict[str, Any]:
        """
        Analizar estructura y patrones de datos de muestra
        
        Returns:
            Análisis completo de la estructura detectada
        """
        try:
            return self._structure_detector.detect_structure(sample_data)
        except Exception as e:
            return {'error': str(e), 'structure': 'unknown'}
    
    def save(self, 
             data: pd.DataFrame, 
             filepath: str, 
             format: Optional[str] = None,
             **kwargs) -> None:
        """
        Guardado inteligente - detecta formato automáticamente si no se especifica
        """
        try:
            self._formatter.auto_save(data, filepath, format, **kwargs)
        except Exception as e:
            raise GenerationError(f"Error guardando datos: {str(e)}")

def generate_data(description_or_request: str,
                 sample_data: Optional[Union[Dict, List, pd.DataFrame, Any]] = None,
                 count: int = 100,
                 format: str = "dataframe",
                 expand_structure: bool = False,
                 creativity_level: float = 0.7,
                 **kwargs) -> Union[pd.DataFrame, Dict, str, List]:
    """
    Función principal para generación dinámica de datos sintéticos
    
    Esta función es completamente flexible y se adapta a cualquier solicitud.
    NO hay estructuras predefinidas - todo es dinámico.
    
    Args:
        description_or_request: Descripción libre de lo que necesitas.
                               Ejemplos:
                               - "Datos de una startup de delivery con pedidos, repartidores y restaurantes"
                               - "Generar dataset para ML de predicción de abandono de clientes"
                               - "Simular datos de red social con usuarios, posts, likes y comentarios"
                               - "Crear datos financieros para backtesting de estrategias"
        
        sample_data: Cualquier muestra de datos (opcional):
                    - DataFrame de pandas
                    - Lista de diccionarios  
                    - Diccionario anidado
                    - JSON string
                    - Incluso archivos CSV/Excel (como path)
        
        count: Cuántos registros generar
        format: Formato de salida ("dataframe", "dict", "json", "csv_string")
        expand_structure: Si True, expande más allá de la muestra original
        creativity_level: Qué tan creativo ser (0.0 = conservador, 1.0 = muy creativo)
        
    Returns:
        Datos en el formato especificado con estructura adaptada dinámicamente
        
    Ejemplos de uso real:
    
        # Caso 1: Generación completamente libre
        data = generate_data(
            "Necesito datos de una empresa de criptomonedas con transacciones, 
             wallets, usuarios, tipos de crypto y todo lo relevante para analytics",
            count=10000,
            creativity_level=0.8
        )
        
        # Caso 2: Expansión desde muestra pequeña
        mini_sample = [
            {"user": "john", "action": "login", "timestamp": "2024-01-01"},
            {"user": "jane", "action": "purchase", "timestamp": "2024-01-02"}
        ]
        
        expanded_data = generate_data(
            "Expandir estos logs incluyendo más tipos de eventos, 
             geolocalización, dispositivos y métricas de performance",
            sample_data=mini_sample,
            count=50000,
            expand_structure=True
        )
        
        # Caso 3: Adaptación inteligente
        sensor_data = pd.read_csv("mi_archivo_sensores.csv")  # cualquier estructura
        more_sensors = generate_data(
            "Generar más datos de sensores pero para diferentes ciudades 
             y con nuevos tipos de mediciones ambientales",
            sample_data=sensor_data,
            count=100000
        )
    """
    
    try:
        # Verificar configuración
        settings = get_settings()
        if not settings and not sample_data:
            raise ConfigurationError(
                "⚠️  Configuración no encontrada. "
                "Configure primero con: configure(openai_api_key='tu-clave')"
            )
        
        # Crear generador y procesar
        generator = SyntheticDataGenerator()
        data = generator.generate(
            description_or_request=description_or_request,
            sample_data=sample_data,
            count=count,
            expand_structure=expand_structure,
            creativity_level=creativity_level,
            **kwargs
        )
        
        # Formatear salida según solicitud
        formatter = DynamicFormatter()
        return formatter.format_output(data, format)
        
    except Exception as e:
        raise GenerationError(f"Error en generate_data: {str(e)}")

def configure(openai_api_key: Optional[str] = None,
             model: str = "gpt-3.5-turbo",
             temperature: float = 0.7,
             **kwargs) -> None:
    """
    Configurar el generador dinámicamente
    
    Args:
        openai_api_key: Tu API key de OpenAI (requerida para generación LLM)
        model: Modelo a usar ("gpt-3.5-turbo", "gpt-4", etc.)
        temperature: Creatividad del modelo (0.0-1.0)
        **kwargs: Otras configuraciones específicas
        
    Ejemplos:
        # Configuración básica
        configure(openai_api_key="sk-tu-clave-aqui")
        
        # Configuración avanzada
        configure(
            openai_api_key="sk-tu-clave-aqui",
            model="gpt-4",
            temperature=0.8,
            sdv_model="CTGAN",
            max_tokens=2000
        )
    """
    
    config_updates = {}
    
    if openai_api_key:
        config_updates['openai_api_key'] = openai_api_key
        config_updates['llm'] = {
            'api_key': openai_api_key,
            'model': model,
            'temperature': temperature
        }
    
    config_updates.update(kwargs)
    
    try:
        update_settings(**config_updates)
        print("✅ Configuración actualizada correctamente")
    except Exception as e:
        raise ConfigurationError(f"Error actualizando configuración: {str(e)}")

def check_setup() -> Dict[str, Any]:
    """
    Verificar que todo esté configurado correctamente para generación dinámica
    
    Returns:
        Estado completo del sistema
    """
    status = {
        "version": __version__,
        "config_status": "❌ No configurado",
        "dependencies": {},
        "capabilities": {
            "llm_generation": False,
            "sdv_generation": False,
            "dynamic_structure": False,
            "multi_format_output": False
        },
        "recommendations": []
    }
    
    # Verificar configuración
    try:
        settings = get_settings()
        if settings.get('openai_api_key') or settings.get('llm', {}).get('api_key'):
            status["config_status"] = "✅ Configurado"
            status["capabilities"]["llm_generation"] = True
        else:
            status["recommendations"].append("Configure API key: configure(openai_api_key='su-clave')")
    except Exception:
        status["recommendations"].append("Configure el sistema: configure(openai_api_key='su-clave')")
    
    # Verificar dependencias
    deps_to_check = {
        "pandas": "pandas",
        "sdv": "sdv", 
        "langchain": "langchain",
        "openai": "openai"
    }
    
    for name, module in deps_to_check.items():
        try:
            imported = __import__(module)
            status["dependencies"][name] = f"✅ {getattr(imported, '__version__', 'installed')}"
            if name == "sdv":
                status["capabilities"]["sdv_generation"] = True
        except ImportError:
            status["dependencies"][name] = "❌ No instalado"
            if name in ["pandas", "openai"]:  # Críticos
                status["recommendations"].append(f"Instalar {name}: pip install {name}")
    
    # Capacidades siempre disponibles
    status["capabilities"]["dynamic_structure"] = True
    status["capabilities"]["multi_format_output"] = True
    
    return status

# Función de ayuda
def examples() -> None:
    """Mostrar ejemplos de uso dinámico"""
    print(f"""
    🎯 Synthetic Data Generator v{__version__} - Ejemplos de Uso Dinámico
    =====================================================================
    
    # 1. GENERACIÓN COMPLETAMENTE LIBRE
    from synthetic_data_generator import generate_data
    
    data = generate_data('''
        Generar datos para una app de fitness que incluya usuarios, 
        entrenamientos, métricas de salud, objetivos personales, 
        progreso temporal y cualquier dato relevante para analytics
    ''', count=5000)
    
    # 2. EXPANSIÓN DESDE MUESTRA MÍNIMA  
    sample = [{"producto": "iPhone", "precio": 999, "stock": 50}]
    
    catalog = generate_data('''
        Expandir este catálogo incluyendo más categorías de productos,
        proveedores, reseñas de clientes, historial de precios,
        y datos de competencia
    ''', sample_data=sample, expand_structure=True, count=10000)
    
    # 3. ADAPTACIÓN INTELIGENTE A CUALQUIER DOMINIO
    log_sample = pd.DataFrame({
        'timestamp': ['2024-01-01 10:00:00'],
        'event': ['user_login'],
        'user_id': [12345]
    })
    
    full_logs = generate_data('''
        Generar logs completos de aplicación incluyendo diferentes
        tipos de eventos, errores, métricas de performance, 
        geolocalización y patrones de uso realistas
    ''', sample_data=log_sample, count=1000000)
    
    # 4. CONFIGURACIÓN Y USO
    from synthetic_data_generator import configure, check_setup
    
    configure(openai_api_key="tu-clave-aqui", creativity_level=0.8)
    print(check_setup())  # Verificar que todo esté listo
    
    ¡El sistema se adapta automáticamente a CUALQUIER tipo de datos!
    """)

# Mensaje de bienvenida
if __name__ == "__main__":
    print(f"🎯 Synthetic Data Generator v{__version__} - Sistema Dinámico")
    print("=" * 60)
    
    status = check_setup()
    print("\n📊 Estado del Sistema:")
    for key, value in status["capabilities"].items():
        icon = "✅" if value else "❌"
        print(f"  {icon} {key.replace('_', ' ').title()}")
    
    if status["recommendations"]:
        print(f"\n💡 Recomendaciones:")
        for rec in status["recommendations"]:
            print(f"  • {rec}")
    
    print(f"\n🚀 Para ver ejemplos: python -c 'from synthetic_data_generator import examples; examples()'")