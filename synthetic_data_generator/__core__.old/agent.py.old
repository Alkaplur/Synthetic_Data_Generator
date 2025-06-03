import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
import warnings
warnings.filterwarnings('ignore')

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from ..utils.visualization import visualize_data
from .models import DataGenerationMode, CustomerVariable, ProductDefinition

class JupyterSyntheticDataAgent:
    """
    Agente de IA optimizado para Jupyter Notebooks
    """
    
    def __init__(self, openai_api_key: str):
        """Inicializa el agente"""
        self.llm = ChatOpenAI(
            openai_api_key=openai_api_key,
            model_name="gpt-3.5-turbo",
            temperature=0.7
        )
        
        # Estado del agente
        self.sample_data = None
        self.metadata = {}
        self.patterns = {}
        self.customer_variables = []
        self.primary_product = None
        self.generated_data = None
        self.generation_mode = None
        
        print("✅ Agente inicializado correctamente")
        print("🔧 Herramientas disponibles:")
        print("   • analyze_sample_data() - Analizar datos existentes")
        print("   • define_variables_interactive() - Definir variables paso a paso")
        print("   • define_product_interactive() - Definir producto paso a paso")
        print("   • generate_data() - Generar datos sintéticos")
        print("   • visualize_data() - Visualizar datos generados")
        print("   • export_data() - Exportar resultados")

    def analyze_sample_data(self, data_source: Union[str, pd.DataFrame], 
                          show_analysis: bool = True) -> Dict:
        """
        Analiza datos de muestra para extraer metadatos y patrones
        
        Args:
            data_source: Ruta al archivo CSV o DataFrame de pandas
            show_analysis: Si mostrar el análisis visual
        """
        print("🔍 Analizando datos de muestra...")
        
        try:
            # Cargar datos
            if isinstance(data_source, str):
                if data_source.endswith('.csv'):
                    self.sample_data = pd.read_csv(data_source)
                elif data_source.endswith('.json'):
                    self.sample_data = pd.read_json(data_source)
                else:
                    raise ValueError("Formato de archivo no soportado")
            elif isinstance(data_source, pd.DataFrame):
                self.sample_data = data_source.copy()
            else:
                raise ValueError("Tipo de datos no soportado")
            
            # Extraer metadatos
            self.metadata = {
                'columns': list(self.sample_data.columns),
                'shape': self.sample_data.shape,
                'dtypes': {col: str(dtype) for col, dtype in self.sample_data.dtypes.items()},
                'null_counts': self.sample_data.isnull().sum().to_dict(),
                'unique_counts': self.sample_data.nunique().to_dict(),
                'memory_usage': self.sample_data.memory_usage(deep=True).sum()
            }
            
            # Extraer patrones
            self.patterns = self._extract_advanced_patterns(self.sample_data)
            self.generation_mode = DataGenerationMode.FROM_SAMPLE
            
            # Mostrar análisis
            if show_analysis:
                self._display_analysis()
            
            # Generar resumen con IA
            summary = self._generate_ai_summary()
            
            print(f"✅ Análisis completado: {self.metadata['shape'][0]} filas, {self.metadata['shape'][1]} columnas")
            print(f"📊 Resumen IA: {summary}")
            
            return {
                'metadata': self.metadata,
                'patterns': self.patterns,
                'ai_summary': summary
            }
            
        except Exception as e:
            print(f"❌ Error al analizar datos: {str(e)}")
            return {}

    def _extract_advanced_patterns(self, data: pd.DataFrame) -> Dict:
        """Extrae patrones avanzados de los datos"""
        patterns = {}
        
        for column in data.columns:
            col_patterns = {
                'column': column,
                'sample_values': data[column].dropna().head(5).tolist()
            }
            
            if data[column].dtype == 'object':
                # Análisis de texto
                non_null = data[column].dropna()
                
                if non_null.str.contains('@', na=False).any():
                    col_patterns['type'] = 'email'
                    domains = non_null.str.extract(r'@(.+)')[0].value_counts()
                    col_patterns['domains'] = domains.head(5).to_dict()
                    
                elif non_null.str.match(r'^\+?[\d\s\-\(\)]+$', na=False).any():
                    col_patterns['type'] = 'phone'
                    
                elif non_null.str.len().mean() > 50:
                    col_patterns['type'] = 'long_text'
                    col_patterns['avg_length'] = non_null.str.len().mean()
                    
                else:
                    col_patterns['type'] = 'categorical'
                    value_counts = non_null.value_counts()
                    col_patterns['unique_values'] = value_counts.head(10).index.tolist()
                    col_patterns['value_distribution'] = value_counts.head(10).to_dict()
                    col_patterns['cardinality'] = len(value_counts)
                    
            elif np.issubdtype(data[column].dtype, np.number):
                non_null = data[column].dropna()
                col_patterns['type'] = 'numerical'
                col_patterns['statistics'] = {
                    'min': float(non_null.min()),
                    'max': float(non_null.max()),
                    'mean': float(non_null.mean()),
                    'median': float(non_null.median()),
                    'std': float(non_null.std()),
                    'quartiles': {
                        'q25': float(non_null.quantile(0.25)),
                        'q75': float(non_null.quantile(0.75))
                    }
                }
                
                # Detectar distribución probable
                if non_null.min() >= 0 and non_null.skew() > 1:
                    col_patterns['probable_distribution'] = 'exponential'
                elif abs(non_null.skew()) < 0.5:
                    col_patterns['probable_distribution'] = 'normal'
                else:
                    col_patterns['probable_distribution'] = 'uniform'
                    
            elif 'datetime' in str(data[column].dtype):
                col_patterns['type'] = 'datetime'
                non_null = pd.to_datetime(data[column], errors='coerce').dropna()
                if len(non_null) > 0:
                    col_patterns['date_range'] = {
                        'min': str(non_null.min()),
                        'max': str(non_null.max())
                    }
            
            patterns[column] = col_patterns
            
        return patterns

    def _display_analysis(self):
        """Muestra análisis visual de los datos"""
        print("\n📊 ANÁLISIS DE DATOS")
        print("=" * 50)
        
        # Información básica
        print(f"📋 Forma del dataset: {self.metadata['shape']}")
        print(f"💾 Uso de memoria: {self.metadata['memory_usage'] / 1024:.2f} KB")
        print(f"🔍 Columnas con valores nulos: {sum(1 for v in self.metadata['null_counts'].values() if v > 0)}")
        
        # Tipos de columnas
        type_counts = {}
        for col, pattern in self.patterns.items():
            col_type = pattern.get('type', 'unknown')
            type_counts[col_type] = type_counts.get(col_type, 0) + 1
        
        print(f"\n📈 Tipos de columnas detectados:")
        for col_type, count in type_counts.items():
            print(f"   • {col_type}: {count} columnas")
        
        # Mostrar sample de datos
        print(f"\n📋 Muestra de datos:")
        print(self.sample_data.head())

    def _generate_ai_summary(self) -> str:
        """Genera un resumen usando IA"""
        summary_prompt = f"""
        Analiza este dataset de clientes y proporciona un resumen conciso en español:
        
        Columnas: {self.metadata['columns']}
        Tipos detectados: {[p.get('type') for p in self.patterns.values()]}
        Filas: {self.metadata['shape'][0]}
        
        Enfócate en:
        1. Qué tipo de clientes representan estos datos
        2. Qué información clave contienen
        3. Qué patrones interesantes observas
        
        Máximo 3 frases.
        """
        
        try:
            response = self.llm([HumanMessage(content=summary_prompt)])
            return response.content
        except:
            return "Datos de clientes con información demográfica y de comportamiento."

    def define_variables_interactive(self):
        """Define variables de cliente de forma interactiva"""
        print("🎯 DEFINICIÓN INTERACTIVA DE VARIABLES")
        print("=" * 50)
        print("Vamos a definir las variables de tus clientes paso a paso.\n")
        
        self.customer_variables = []
        
        while True:
            print(f"\n📝 Variable #{len(self.customer_variables) + 1}")
            
            # Nombre de la variable
            name = input("Nombre de la variable: ").strip()
            if not name:
                break
                
            # Tipo de dato
            print("\nTipos disponibles:")
            print("1. categorical - Opciones limitadas (ej: género, ciudad)")
            print("2. numerical - Números (ej: edad, ingresos)")
            print("3. text - Texto libre (ej: nombres, comentarios)")
            print("4. email - Direcciones de email")
            print("5. phone - Números de teléfono")
            print("6. date - Fechas")
            
            type_choice = input("Selecciona el tipo (1-6): ").strip()
            type_mapping = {
                '1': 'categorical', '2': 'numerical', '3': 'text',
                '4': 'email', '5': 'phone', '6': 'date'
            }
            data_type = type_mapping.get(type_choice, 'text')
            
            # Descripción
            description = input("Descripción breve: ").strip()
            
            # Configuración específica por tipo
            variable = CustomerVariable(name=name, data_type=data_type, description=description)
            
            if data_type == 'categorical':
                values_input = input("Valores posibles (separados por coma): ")
                variable.possible_values = [v.strip() for v in values_input.split(',')]
                
            elif data_type == 'numerical':
                min_val = float(input("Valor mínimo: ") or 0)
                max_val = float(input("Valor máximo: ") or 100)
                variable.min_value = min_val
                variable.max_value = max_val
                
                dist_choice = input("Distribución (1=uniforme, 2=normal): ").strip()
                variable.distribution = 'normal' if dist_choice == '2' else 'uniform'
            
            self.customer_variables.append(variable)
            print(f"✅ Variable '{name}' agregada")
            
            # Continuar?
            continue_choice = input("\n¿Agregar otra variable? (s/n): ").strip().lower()
            if continue_choice != 's':
                break
        
        self.generation_mode = DataGenerationMode.FROM_DEFINITION
        print(f"\n✅ {len(self.customer_variables)} variables definidas correctamente")

    def define_product_interactive(self):
        """Define el producto principal de forma interactiva"""
        print("\n🛍️ DEFINICIÓN DEL PRODUCTO PRINCIPAL")
        print("=" * 40)
        
        name = input("Nombre del producto: ").strip()
        category = input("Categoría del producto: ").strip()
        
        min_price = float(input("Precio mínimo: ") or 0)
        max_price = float(input("Precio máximo: ") or 100)
        
        features_input = input("Características (separadas por coma): ")
        features = [f.strip() for f in features_input.split(',')]
        
        prob = float(input("Probabilidad de compra (0.0-1.0, default 0.8): ") or 0.8)
        
        self.primary_product = ProductDefinition(
            name=name,
            category=category,
            price_range=(min_price, max_price),
            features=features,
            purchase_probability=prob
        )
        
        print(f"✅ Producto '{name}' definido correctamente")

    def generate_data(self, num_records: int = 1000, seed: int = 42) -> pd.DataFrame:
        """
        Genera datos sintéticos
        
        Args:
            num_records: Número de registros a generar
            seed: Semilla para reproducibilidad
        """
        print(f"🎲 Generando {num_records} registros sintéticos...")
        
        np.random.seed(seed)
        
        try:
            if self.generation_mode == DataGenerationMode.FROM_SAMPLE:
                self.generated_data = self._generate_from_sample_advanced(num_records)
            elif self.generation_mode == DataGenerationMode.FROM_DEFINITION:
                self.generated_data = self._generate_from_definition_advanced(num_records)
            else:
                raise ValueError("Debe analizar datos o definir variables primero")
            
            print(f"✅ Datos generados: {len(self.generated_data)} filas, {len(self.generated_data.columns)} columnas")
            return self.generated_data
            
        except Exception as e:
            print(f"❌ Error al generar datos: {str(e)}")
            return pd.DataFrame()

    def _generate_from_sample_advanced(self, num_records: int) -> pd.DataFrame:
        """Generación avanzada desde datos de muestra"""
        synthetic_data = {}
        
        for column in self.sample_data.columns:
            pattern = self.patterns.get(column, {})
            col_type = pattern.get('type', 'categorical')
            
            if col_type == 'categorical':
                # Mantener distribución original
                value_counts = self.sample_data[column].value_counts(normalize=True)
                synthetic_data[column] = np.random.choice(
                    value_counts.index, 
                    size=num_records, 
                    p=value_counts.values
                )
                
            elif col_type == 'numerical':
                stats = pattern.get('statistics', {})
                distribution = pattern.get('probable_distribution', 'normal')
                
                if distribution == 'normal':
                    values = np.random.normal(
                        stats.get('mean', 0), 
                        stats.get('std', 1), 
                        num_records
                    )
                    # Aplicar límites observados
                    values = np.clip(values, stats.get('min', values.min()), stats.get('max', values.max()))
                    
                elif distribution == 'exponential':
                    # Distribución exponencial ajustada
                    scale = stats.get('mean', 1)
                    values = np.random.exponential(scale, num_records)
                    values = np.clip(values, stats.get('min', 0), stats.get('max', values.max()))
                    
                else:  # uniform
                    values = np.random.uniform(
                        stats.get('min', 0),
                        stats.get('max', 100),
                        num_records
                    )
                
                synthetic_data[column] = values
                
            elif col_type == 'email':
                domains = pattern.get('domains', {'gmail.com': 1})
                domain_list = list(domains.keys())
                domain_weights = [domains[d] for d in domain_list]
                domain_weights = np.array(domain_weights) / sum(domain_weights)
                
                names = [f"user{i:05d}" for i in range(num_records)]
                chosen_domains = np.random.choice(domain_list, num_records, p=domain_weights)
                synthetic_data[column] = [f"{name}@{domain}" for name, domain in zip(names, chosen_domains)]
                
            elif col_type == 'phone':
                # Generar teléfonos sintéticos
                synthetic_data[column] = [f"+34-6{np.random.randint(10000000, 99999999)}" for _ in range(num_records)]
                
            elif col_type == 'datetime':
                date_range = pattern.get('date_range', {})
                if date_range:
                    start_date = pd.to_datetime(date_range.get('min', '2020-01-01'))
                    end_date = pd.to_datetime(date_range.get('max', '2024-12-31'))
                    
                    time_diff = (end_date - start_date).days
                    random_days = np.random.randint(0, time_diff, num_records)
                    synthetic_data[column] = [start_date + timedelta(days=int(d)) for d in random_days]
                
            else:  # text u otros
                unique_values = self.sample_data[column].dropna().unique()
                if len(unique_values) > 0:
                    synthetic_data[column] = np.random.choice(unique_values, num_records)
                else:
                    synthetic_data[column] = [f"{column}_{i}" for i in range(num_records)]
        
        return pd.DataFrame(synthetic_data)

    def _generate_from_definition_advanced(self, num_records: int) -> pd.DataFrame:
        """Generación avanzada desde definiciones"""
        synthetic_data = {}
        
        for variable in self.customer_variables:
            if variable.data_type == 'categorical':
                synthetic_data[variable.name] = np.random.choice(
                    variable.possible_values, 
                    num_records
                )
                
            elif variable.data_type == 'numerical':
                if variable.distribution == 'normal':
                    mean = (variable.min_value + variable.max_value) / 2
                    std = (variable.max_value - variable.min_value) / 6  # Aprox 99.7% dentro del rango
                    values = np.random.normal(mean, std, num_records)
                    values = np.clip(values, variable.min_value, variable.max_value)
                else:  # uniform
                    values = np.random.uniform(
                        variable.min_value, 
                        variable.max_value, 
                        num_records
                    )
                synthetic_data[variable.name] = values
                
            elif variable.data_type == 'email':
                domains = ['gmail.com', 'yahoo.com', 'outlook.com', 'empresa.com']
                names = [f"cliente{i:05d}" for i in range(num_records)]
                synthetic_data[variable.name] = [
                    f"{name}@{np.random.choice(domains)}" 
                    for name in names
                ]
                
            elif variable.data_type == 'phone':
                synthetic_data[variable.name] = [
                    f"+34-6{np.random.randint(10000000, 99999999)}" 
                    for _ in range(num_records)
                ]
                
            elif variable.data_type == 'date':
                start_date = datetime.now() - timedelta(days=365*2)
                synthetic_data[variable.name] = [
                    start_date + timedelta(days=np.random.randint(0, 730))
                    for _ in range(num_records)
                ]
                
            else:  # text
                synthetic_data[variable.name] = [
                    f"{variable.name}_{i:05d}" for i in range(num_records)
                ]
        
        # Agregar producto principal si está definido
        if self.primary_product:
            # Determinar quién compra el producto
            purchases = np.random.random(num_records) < self.primary_product.purchase_probability
            
            synthetic_data['compro_producto'] = purchases
            synthetic_data['producto_comprado'] = [
                self.primary_product.name if purchased else 'Ninguno'
                for purchased in purchases
            ]
            synthetic_data['categoria_producto'] = [
                self.primary_product.category if purchased else 'N/A'
                for purchased in purchases
            ]
            synthetic_data['precio_pagado'] = [
                np.random.uniform(*self.primary_product.price_range) if purchased else 0
                for purchased in purchases
            ]
        
        return pd.DataFrame(synthetic_data)

    def visualize_data(self, columns: Optional[List[str]] = None):
        """Visualiza los datos generados"""
        if self.generated_data is None:
            print("❌ No hay datos generados para visualizar")
            return
        
        visualize_data(self.generated_data, columns)

    def export_data(self, filename: str, format_type: str = 'csv'):
        """
        Exporta los datos generados
        
        Args:
            filename: Nombre del archivo
            format_type: 'csv', 'json', 'excel'
        """
        if self.generated_data is None:
            print("❌ No hay datos generados para exportar")
            return
        
        try:
            if format_type.lower() == 'csv':
                self.generated_data.to_csv(filename, index=False)
            elif format_type.lower() == 'json':
                self.generated_data.to_json(filename, orient='records', indent=2)
            elif format_type.lower() == 'excel':
                self.generated_data.to_excel(filename, index=False)
            else:
                print(f"❌ Formato no soportado: {format_type}")
                return
            
            print(f"✅ Datos exportados a {filename} en formato {format_type}")
            print(f"📊 {len(self.generated_data)} registros, {len(self.generated_data.columns)} columnas")
            
        except Exception as e:
            print(f"❌ Error al exportar: {str(e)}")

    def get_summary(self) -> Dict:
        """Obtiene un resumen completo del proceso"""
        summary = {
            'generation_mode': self.generation_mode.value if self.generation_mode else None,
            'sample_data_shape': self.metadata.get('shape') if self.sample_data is not None else None,
            'variables_defined': len(self.customer_variables),
            'product_defined': self.primary_product is not None,
            'data_generated': self.generated_data is not None,
            'generated_records': len(self.generated_data) if self.generated_data is not None else 0
        }
        return summary 