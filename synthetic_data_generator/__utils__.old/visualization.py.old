import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import List, Optional

def visualize_data(data: pd.DataFrame, columns: Optional[List[str]] = None):
    """Visualiza los datos generados"""
    if data is None:
        print("❌ No hay datos generados para visualizar")
        return
    
    print("📊 VISUALIZACIÓN DE DATOS SINTÉTICOS")
    print("=" * 40)
    
    # Seleccionar columnas
    if columns is None:
        columns = data.columns[:6]  # Primeras 6 columnas
    
    # Crear subplots
    n_cols = min(3, len(columns))
    n_rows = (len(columns) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    if n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    else:
        axes = axes.flatten()
    
    for i, column in enumerate(columns):
        ax = axes[i] if len(columns) > 1 else axes
        
        if data[column].dtype in ['object', 'bool']:
            # Gráfico de barras para categóricas
            value_counts = data[column].value_counts().head(10)
            ax.bar(range(len(value_counts)), value_counts.values)
            ax.set_xticks(range(len(value_counts)))
            ax.set_xticklabels(value_counts.index, rotation=45, ha='right')
            ax.set_title(f'{column} (Categórica)')
            
        else:
            # Histograma para numéricas
            ax.hist(data[column].dropna(), bins=30, alpha=0.7)
            ax.set_title(f'{column} (Numérica)')
        
        ax.grid(True, alpha=0.3)
    
    # Ocultar subplots vacíos
    for i in range(len(columns), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()
    
    # Estadísticas resumidas
    print("\n📈 Estadísticas resumidas:")
    print(data.describe(include='all')) 