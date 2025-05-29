"""
Generador de datos sintéticos usando SDV (Synthetic Data Vault)
"""

import pandas as pd
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata
from sdv.evaluation.single_table import evaluate_quality

class SDVGenerator:
    def __init__(self):
        self.synthesizer = None
        self.metadata = None
        self.original_data = None
        
    def train_from_sample(self, sample_data: pd.DataFrame) -> dict:
        """
        Entrena el modelo SDV con datos de muestra
        
        Args:
            sample_data: DataFrame con datos de muestra
            
        Returns:
            dict: Resultado del entrenamiento
        """
        try:
            # Guardar datos originales para evaluación
            self.original_data = sample_data.copy()
            
            # Configurar metadata
            self.metadata = SingleTableMetadata()
            self.metadata.detect_from_dataframe(sample_data)
            
            # Crear y entrenar sintetizador
            self.synthesizer = GaussianCopulaSynthesizer(self.metadata)
            self.synthesizer.fit(sample_data)
            
            return {
                "success": True,
                "message": "Modelo SDV entrenado exitosamente",
                "metadata": {
                    "columns": {col: info.get('sdtype', 'unknown') 
                              for col, info in self.metadata.columns.items()},
                    "num_rows": len(sample_data)
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Error entrenando modelo SDV: {str(e)}"
            }
    
    def generate_data(self, num_rows: int = 100) -> dict:
        """
        Genera datos sintéticos usando el modelo entrenado
        
        Args:
            num_rows: Número de registros a generar
            
        Returns:
            dict: Datos sintéticos generados
        """
        try:
            if not self.synthesizer:
                return {
                    "success": False,
                    "error": "El modelo no está entrenado. Llama a train_from_sample primero."
                }
            
            # Generar datos sintéticos
            synthetic_data = self.synthesizer.sample(num_rows=num_rows)
            
            # Evaluar calidad
            quality_report = evaluate_quality(
                real_data=self.original_data,
                synthetic_data=synthetic_data,
                metadata=self.metadata
            )
            
            return {
                "success": True,
                "data": synthetic_data,
                "quality_score": quality_report.get_score(),
                "metadata": {
                    "num_rows": len(synthetic_data),
                    "columns": list(synthetic_data.columns)
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Error generando datos sintéticos: {str(e)}"
            }
    
    def save_model(self, filepath: str) -> dict:
        """
        Guarda el modelo entrenado
        
        Args:
            filepath: Ruta donde guardar el modelo
            
        Returns:
            dict: Resultado de la operación
        """
        try:
            if not self.synthesizer:
                return {
                    "success": False,
                    "error": "No hay modelo para guardar"
                }
            
            self.synthesizer.save(filepath)
            return {
                "success": True,
                "message": f"Modelo guardado en {filepath}"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Error guardando modelo: {str(e)}"
            }
    
    def load_model(self, filepath: str) -> dict:
        """
        Carga un modelo guardado
        
        Args:
            filepath: Ruta del modelo a cargar
            
        Returns:
            dict: Resultado de la operación
        """
        try:
            self.synthesizer = GaussianCopulaSynthesizer.load(filepath)
            self.metadata = self.synthesizer.get_metadata()
            
            return {
                "success": True,
                "message": "Modelo cargado exitosamente",
                "metadata": {
                    "columns": {col: info.get('sdtype', 'unknown') 
                              for col, info in self.metadata.columns.items()}
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Error cargando modelo: {str(e)}"
            } 