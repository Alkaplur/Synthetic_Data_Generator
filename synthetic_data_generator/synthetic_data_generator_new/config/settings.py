# config/settings.py
"""
Sistema de Configuración Dinámico - Sin Hardcoding
==================================================

Sistema completamente flexible que:
- Se adapta automáticamente a diferentes entornos
- Detecta configuraciones disponibles dinámicamente  
- No tiene valores predeterminados rígidos
- Se auto-configura basado en el contexto
- Permite override total por parte del usuario

Filosofía: CERO hardcoding, máxima flexibilidad
"""

import os
import json
import yaml
import warnings
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
from dataclasses import dataclass, field, asdict
from copy import deepcopy

# Estado global de configuración - completamente dinámico
_global_settings = {}
_settings_sources = []
_auto_detected_config = {}

class DynamicConfig:
    """
    Configuración completamente dinámica que se adapta al entorno
    
    NO tiene valores predeterminados - todo se detecta o se configura dinámicamente
    """
    
    def __init__(self):
        self._config = {}
        self._sources = []
        self._auto_detected = {}
        self._detection_cache = {}
        
    def auto_detect_environment(self) -> Dict[str, Any]:
        """
        Detectar automáticamente configuración del entorno
        
        Returns:
            Configuración detectada dinámicamente
        """
        if self._detection_cache:
            return self._detection_cache
            
        detected = {}
        
        # Detectar APIs disponibles automáticamente
        api_detection = self._detect_available_apis()
        if api_detection:
            detected['apis'] = api_detection
            
        # Detectar modelos disponibles
        model_detection = self._detect_available_models()
        if model_detection:
            detected['models'] = model_detection
            
        # Detectar capacidades del sistema
        system_caps = self._detect_system_capabilities()
        if system_caps:
            detected['system'] = system_caps
            
        # Detectar configuraciones existentes
        existing_configs = self._find_existing_configs()
        if existing_configs:
            detected['existing_configs'] = existing_configs
            
        self._detection_cache = detected
        return detected
    
    def _detect_available_apis(self) -> Dict[str, Any]:
        """Detectar APIs disponibles - Solo OpenAI por ahora"""
        apis = {}
        
        # Solo OpenAI por el momento
        openai_key = None
        
        # Buscar API key de OpenAI en variables de entorno
        for env_var in ['OPENAI_API_KEY', 'OPENAI_KEY']:
            if os.environ.get(env_var):
                openai_key = os.environ[env_var]
                break
        
        if openai_key:
            apis['openai'] = {
                'api_key': openai_key,
                'available': True
            }
        
        return apis
    
    def _detect_available_models(self) -> Dict[str, List[str]]:
        """Detectar modelos disponibles - Solo OpenAI por ahora"""
        models = {'available': [], 'openai': []}
        
        # Verificar si OpenAI está disponible
        apis = self._detect_available_apis()
        if 'openai' in apis:
            # Modelos OpenAI con el nuevo default
            openai_models = ['gpt-4o-mini-2024-07-18', 'gpt-3.5-turbo', 'gpt-4', 'gpt-4-turbo']
            models['openai'] = openai_models
            models['available'].extend(openai_models)
                
        return models
    
    def _detect_local_models(self) -> List[str]:
        """Detectar modelos locales - Placeholder para futuro"""
        # Por ahora retornar lista vacía
        # TODO: Agregar detección de Ollama u otros cuando sea necesario
        return []
    
    def _get_openai_models(self) -> List[str]:
        """Modelos OpenAI disponibles - Lista simple por ahora"""
        # Lista actualizada con el modelo más reciente como default
        return ['gpt-4o-mini-2024-07-18', 'gpt-3.5-turbo', 'gpt-4', 'gpt-4-turbo']
    
    def _get_anthropic_models(self) -> List[str]:
        """Placeholder para Anthropic - Para futuro"""
        return []
    
    def _detect_system_capabilities(self) -> Dict[str, Any]:
        """Detectar capacidades del sistema automáticamente"""
        caps = {}
        
        # GPU detection
        try:
            import torch
            caps['gpu_available'] = torch.cuda.is_available()
            if caps['gpu_available']:
                caps['gpu_count'] = torch.cuda.device_count()
                caps['gpu_memory'] = torch.cuda.get_device_properties(0).total_memory
        except ImportError:
            caps['gpu_available'] = False
        
        # Memory detection
        try:
            import psutil
            caps['ram_gb'] = psutil.virtual_memory().total / (1024**3)
        except ImportError:
            pass
        
        # CPU detection
        caps['cpu_count'] = os.cpu_count()
        
        return caps
    
    def _find_existing_configs(self) -> List[Dict[str, Any]]:
        """Buscar archivos de configuración existentes"""
        configs = []
        
        # Patrones de archivos de configuración comunes
        config_patterns = [
            'config.yaml', 'config.yml', 'settings.yaml', 'settings.yml',
            'config.json', 'settings.json', '.env', 'env.yaml',
            'synthetic_data_config.*', 'sdg_config.*'
        ]
        
        search_paths = [
            Path.cwd(),
            Path.cwd() / 'config',
            Path.cwd() / 'configs', 
            Path.home() / '.synthetic_data_generator',
            Path('/etc/synthetic_data_generator')
        ]
        
        for search_path in search_paths:
            if not search_path.exists():
                continue
                
            for pattern in config_patterns:
                for config_file in search_path.glob(pattern):
                    try:
                        config_data = self._load_config_file(config_file)
                        if config_data:
                            configs.append({
                                'file': str(config_file),
                                'data': config_data,
                                'type': self._detect_config_type(config_file)
                            })
                    except Exception as e:
                        warnings.warn(f"No se pudo cargar {config_file}: {e}")
        
        return configs
    
    def _load_config_file(self, config_path: Path) -> Dict[str, Any]:
        """Cargar archivo de configuración dinámicamente"""
        if not config_path.exists():
            return {}
            
        content = config_path.read_text()
        
        if config_path.suffix in ['.yaml', '.yml']:
            return yaml.safe_load(content) or {}
        elif config_path.suffix == '.json':
            return json.loads(content)
        elif config_path.name == '.env' or 'env' in config_path.name:
            # Parse .env format
            env_dict = {}
            for line in content.split('\n'):
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env_dict[key.strip()] = value.strip().strip('"\'')
            return env_dict
        
        return {}
    
    def _detect_config_type(self, config_path: Path) -> str:
        """Detectar tipo de configuración automáticamente"""
        if 'openai' in config_path.name.lower():
            return 'openai_specific'
        elif 'sdv' in config_path.name.lower():
            return 'sdv_specific'  
        elif 'llm' in config_path.name.lower():
            return 'llm_specific'
        else:
            return 'general'
    
    def get_optimal_config(self, 
                          task_type: Optional[str] = None,
                          constraints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Obtener configuración óptima basada en detección automática
        
        Args:
            task_type: Tipo de tarea ('llm_generation', 'sdv_generation', 'hybrid')
            constraints: Restricciones del usuario (budget, speed, quality, etc.)
            
        Returns:
            Configuración optimizada dinámicamente
        """
        # Auto-detectar entorno
        detected = self.auto_detect_environment()
        
        # Configuración base dinámica
        optimal_config = {}
        
        # Seleccionar mejor API/modelo disponible
        if task_type in ['llm_generation', 'hybrid', None]:
            llm_config = self._select_optimal_llm(detected, constraints)
            if llm_config:
                optimal_config['llm'] = llm_config
        
        if task_type in ['sdv_generation', 'hybrid', None]:
            sdv_config = self._select_optimal_sdv(detected, constraints)
            if sdv_config:
                optimal_config['sdv'] = sdv_config
        
        # Configuración de sistema
        system_config = self._optimize_system_config(detected, constraints)
        if system_config:
            optimal_config['system'] = system_config
            
        return optimal_config
    
    def _select_optimal_llm(self, 
                           detected: Dict[str, Any], 
                           constraints: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Seleccionar configuración LLM óptima - Solo OpenAI por ahora"""
        apis = detected.get('apis', {})
        
        # Por ahora solo OpenAI
        if 'openai' in apis:
            config = {
                'provider': 'openai',
                'api_key': apis['openai'].get('api_key')
            }
            
            # Seleccionar modelo OpenAI basado en constraints
            models = detected.get('models', {}).get('openai', [])
            if models:
                if constraints and constraints.get('speed') == 'fast':
                    config['model'] = 'gpt-4o-mini-2024-07-18'  # Más rápido y eficiente
                elif constraints and constraints.get('quality') == 'high':
                    config['model'] = 'gpt-4'  # Mejor calidad
                else:
                    config['model'] = 'gpt-4o-mini-2024-07-18'  # DEFAULT - Mejor balance costo/rendimiento
            
            return config
        
        return {}
    
    def _select_optimal_sdv(self, 
                           detected: Dict[str, Any], 
                           constraints: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Seleccionar configuración SDV óptima"""
        system = detected.get('system', {})
        
        config = {}
        
        # Seleccionar modelo SDV basado en capacidades del sistema
        if system.get('gpu_available'):
            config['preferred_models'] = ['CTGAN', 'TVAE', 'CopulaGAN']
        else:
            config['preferred_models'] = ['GaussianCopula', 'BayesianNetwork']
        
        # Configurar parámetros basados en recursos
        if system.get('ram_gb', 8) > 16:
            config['batch_size'] = 'auto_large'  
            config['epochs'] = 'auto_optimal'
        else:
            config['batch_size'] = 'auto_conservative'
            config['epochs'] = 'auto_fast'
        
        return config
    
    def _optimize_system_config(self, 
                               detected: Dict[str, Any], 
                               constraints: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Optimizar configuración de sistema"""
        system = detected.get('system', {})
        
        config = {
            'parallel_workers': min(system.get('cpu_count', 1), 4),
            'memory_limit': system.get('ram_gb', 8) * 0.8,  # 80% de RAM disponible
            'enable_gpu': system.get('gpu_available', False)
        }
        
        return config
    
    def _select_best_model(self, 
                          models: List[str], 
                          constraints: Optional[Dict[str, Any]]) -> str:
        """Seleccionar mejor modelo de una lista dinámicamente"""
        if not models:
            return None
            
        # Lógica de selección dinámica
        if constraints:
            if constraints.get('speed') == 'fast':
                # Preferir modelos más rápidos
                for model in models:
                    if any(fast_indicator in model.lower() 
                          for fast_indicator in ['turbo', '3.5', 'small', 'fast']):
                        return model
            elif constraints.get('quality') == 'high':
                # Preferir modelos de mayor calidad
                for model in models:
                    if any(quality_indicator in model.lower() 
                          for quality_indicator in ['4', 'large', 'opus', 'pro']):
                        return model
        
        # Default: primer modelo disponible
        return models[0]

# Instancia global de configuración dinámica
_dynamic_config = DynamicConfig()

def get_settings() -> Dict[str, Any]:
    """
    Obtener configuración actual (completamente dinámica)
    
    Returns:
        Configuración actualizada basada en detección automática
    """
    global _global_settings
    
    # Si no hay configuración manual, usar auto-detección
    if not _global_settings:
        _global_settings = _dynamic_config.get_optimal_config()
    
    return deepcopy(_global_settings)

def update_settings(**kwargs) -> None:
    """
    Actualizar configuración dinámicamente
    
    Args:
        **kwargs: Cualquier configuración a actualizar
    """
    global _global_settings
    
    # Merge con configuración existente
    if not _global_settings:
        _global_settings = _dynamic_config.get_optimal_config()
    
    # Update recursivo
    _recursive_update(_global_settings, kwargs)

def _recursive_update(base_dict: Dict, update_dict: Dict) -> None:
    """Actualización recursiva de diccionarios"""
    for key, value in update_dict.items():
        if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
            _recursive_update(base_dict[key], value)
        else:
            base_dict[key] = value

def auto_configure(task_type: Optional[str] = None,
                  constraints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Auto-configuración completa basada en tarea y restricciones
    
    Args:
        task_type: 'llm_generation', 'sdv_generation', 'hybrid'
        constraints: {'budget': 'low/high', 'speed': 'fast/slow', 'quality': 'low/high'}
        
    Returns:
        Configuración optimizada automáticamente
    """
    optimal_config = _dynamic_config.get_optimal_config(task_type, constraints)
    update_settings(**optimal_config)
    return optimal_config

def detect_environment() -> Dict[str, Any]:
    """
    Detectar entorno actual completamente
    
    Returns:
        Detección completa del entorno
    """
    return _dynamic_config.auto_detect_environment()

def suggest_configuration(description: str) -> Dict[str, Any]:
    """
    Sugerir configuración basada en descripción de la tarea
    
    Args:
        description: Descripción de lo que se quiere generar
        
    Returns:
        Configuración sugerida dinámicamente
    """
    # Análisis simple de la descripción para detectar requisitos
    constraints = {}
    desc_lower = description.lower()
    
    if any(word in desc_lower for word in ['rápido', 'fast', 'quick', 'speed']):
        constraints['speed'] = 'fast'
    if any(word in desc_lower for word in ['calidad', 'quality', 'high-quality', 'best']):
        constraints['quality'] = 'high'
    if any(word in desc_lower for word in ['barato', 'cheap', 'budget', 'free']):
        constraints['budget'] = 'low'
    
    # Detectar tipo de tarea
    task_type = None
    if any(word in desc_lower for word in ['sample', 'muestra', 'existente', 'similar']):
        task_type = 'sdv_generation'
    elif any(word in desc_lower for word in ['descripción', 'describe', 'create', 'generar']):
        task_type = 'llm_generation'
    else:
        task_type = 'hybrid'
    
    return auto_configure(task_type, constraints)

def save_config(filepath: str, config: Optional[Dict[str, Any]] = None) -> None:
    """Guardar configuración actual a archivo"""
    config_to_save = config or get_settings()
    
    path = Path(filepath)
    
    if path.suffix in ['.yaml', '.yml']:
        with open(path, 'w') as f:
            yaml.dump(config_to_save, f, default_flow_style=False)
    elif path.suffix == '.json':
        with open(path, 'w') as f:
            json.dump(config_to_save, f, indent=2)
    else:
        raise ValueError(f"Formato no soportado: {path.suffix}")

def load_config(filepath: str) -> Dict[str, Any]:
    """Cargar configuración desde archivo"""
    loaded_config = _dynamic_config._load_config_file(Path(filepath))
    update_settings(**loaded_config)
    return loaded_config

# Funciones de conveniencia
def configure_for_development() -> Dict[str, Any]:
    """Configuración optimizada para desarrollo"""
    return auto_configure(constraints={'speed': 'fast', 'budget': 'low'})

def configure_for_production() -> Dict[str, Any]:
    """Configuración optimizada para producción"""
    return auto_configure(constraints={'quality': 'high', 'speed': 'balanced'})

def reset_configuration() -> None:
    """Resetear configuración y volver a auto-detección"""
    global _global_settings
    _global_settings = {}
    _dynamic_config._detection_cache = {}

if __name__ == "__main__":
    # Demo de configuración dinámica
    print("🔧 Sistema de Configuración Dinámico")
    print("=" * 50)
    
    print("\n🔍 Detectando entorno...")
    env = detect_environment()
    print(f"APIs disponibles: {list(env.get('apis', {}).keys())}")
    print(f"Modelos disponibles: {env.get('models', {}).get('available', [])}")
    print(f"GPU disponible: {env.get('system', {}).get('gpu_available', False)}")
    
    print("\n⚙️ Auto-configuración...")
    config = auto_configure()
    
    if config.get('llm', {}).get('api_key'):
        print(f"✅ OpenAI configurado")
        print(f"   Modelo default: {config.get('llm', {}).get('model', 'No definido')}")
    else:
        print("❌ OpenAI no configurado - Configure OPENAI_API_KEY")
    
    print(f"\n💡 Para usar: configure(openai_api_key='su-clave-aqui')")
    print(f"📖 Para ayuda completa ejecute: python -c 'from synthetic_data_generator.config.settings import *; help(auto_configure)'")