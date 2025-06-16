"""
Pure Synthetic Tools - Versión simplificada para agentes con NVIDIA API
"""

import os
import pandas as pd
import openai
from typing import Dict, Any

from agents import function_tool, RunContextWrapper
from tools import SyntheticDataContext

# Configurar cliente NVIDIA (compatible con OpenAI)
nvidia_client = openai.OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="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"
)

def call_nvidia_nemotron(prompt, max_tokens=1000):
    """Llamar a Nemotron 70B via NVIDIA API"""
    try:
        response = nvidia_client.chat.completions.create(
            model="nvidia/llama-3.1-nemotron-70b-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=max_tokens,
            stream=False
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"🔥 Error llamando NVIDIA API: {e}")
        return f"Error: {str(e)}"

def clean_csv_response(response: str) -> str:
    """Limpiar respuesta del LLM para obtener CSV puro"""
    lines = response.split('\n')
    csv_lines = []
    
    for line in lines:
        line = line.strip()
        
        # Quitar caracteres problemáticos
        line = line.replace('\\', '')  # Quitar \_ 
        line = line.replace('```csv', '')  # Quitar markdown
        line = line.replace('```', '')  # Quitar markdown
        
        # Es línea de headers?
        if (line.startswith('id,') and 
            ('name,' in line or 'first_name,' in line)):
            csv_lines.append(line)
        # Es línea de datos válida? 
        elif (line and 
              len(line) > 3 and
              line[0].isdigit() and 
              line[1] == ',' and
              line.count(',') >= 5 and  # Mínimo 6 columnas
              not '[then' in line and  # Excluir "[then 7 more data rows]"
              not 'more' in line):
            csv_lines.append(line)
    
    return '\n'.join(csv_lines)

@function_tool
def generate_synthetic_data_simple(
    wrapper: RunContextWrapper[SyntheticDataContext],
    data_type: str,
    num_rows: int
) -> Dict[str, Any]:
    """
    Genera datos sintéticos usando Nemotron 70B via NVIDIA API.
    
    Args:
        data_type: Tipo de datos (customers, products, employees, etc)
        num_rows: Número de filas a generar
        
    Returns:
        Información del archivo generado
    """
    try:
        context = wrapper.context
        
        print(f"🔧 DEBUG: Iniciando generación de {num_rows} {data_type}")
        
        # Validaciones
        if num_rows <= 0 or num_rows > 100:
            print(f"❌ DEBUG: Validación falló - num_rows: {num_rows}")
            return {
                "success": False,
                "error": f"Número de filas debe estar entre 1 y 100, recibido: {num_rows}"
            }
        
        print("🔧 DEBUG: Generando prompt...")
        
        # Prompts específicos por tipo de dato
        prompts = {
            "customers": f"""Generate exactly {num_rows} customer records in CSV format.
Start with headers: id,first_name,last_name,email,age,city,country
Then generate exactly {num_rows} data rows with Spanish names and cities.
Output only CSV data, no explanations.""",

            "products": f"""Generate exactly {num_rows} product records in CSV format.
Start with headers: id,name,category,price,brand,stock
Then generate exactly {num_rows} data rows with Spanish product names.
Output only CSV data, no explanations.""",

            "employees": f"""Generate exactly {num_rows} employee records in CSV format.
Start with headers: id,first_name,last_name,department,position,salary,hire_date
Then generate exactly {num_rows} data rows with Spanish names and realistic departments.
Output only CSV data, no explanations."""
        }
        
        prompt = prompts.get(data_type, f"Generate {num_rows} rows of {data_type} data in CSV format with headers:")
        
        print("🔧 DEBUG: Llamando NVIDIA API...")
        
        # Generar datos con NVIDIA API
        response = call_nvidia_nemotron(prompt, 100000)
        
        print(f"✅ DEBUG: Respuesta recibida: {len(response)} caracteres")
        
        # DEBUG: Guardar respuesta cruda para ver qué genera
        debug_filename = f"DEBUG_{data_type}_{num_rows}.txt"
        with open(debug_filename, 'w') as f:
            f.write(response)
        print(f"📁 DEBUG: Respuesta guardada en {debug_filename}")
        
        # Limpiar respuesta
        clean_csv = clean_csv_response(response)
        
        if not clean_csv or len(clean_csv.strip()) < 10:
            print("⚠️ DEBUG: CSV limpio está vacío, usando respuesta cruda")
            clean_csv = response
        
        # Directorio de salida específico
        output_dir = "/Users/davidnogueras/Desktop/Cursor/Synthetic_Data_Generator/synthetic_data_generator_basic/Sample data"

        # Crear directorio si no existe
        os.makedirs(output_dir, exist_ok=True)

        # Crear nombre de archivo con ruta completa
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{data_type}_synthetic_{num_rows}rows_{timestamp}.csv"
        full_path = os.path.join(output_dir, filename)

        # Guardar archivo en la carpeta específica
        with open(full_path, 'w') as f:
            f.write(clean_csv)

        print(f"📁 DEBUG: Archivo guardado en: {full_path}")
        
        # Actualizar contexto
        context.generated_file_path = full_path
        context.generated_file_id = timestamp
        context.generated_rows = num_rows
        context.last_model_used = "nvidia/llama-3.1-nemotron-70b-instruct"
        
        return {
            "success": True,
            "filename": filename,
            "rows_generated": num_rows,
            "data_type": data_type,
            "file_path": full_path,
            "message": f"✅ Generados {num_rows} registros de {data_type} usando Nemotron 70B"
        }
        
    except Exception as e:
        print(f"🔥 DEBUG ERROR: {str(e)}")
        import traceback
        print(f"🔥 DEBUG TRACEBACK: {traceback.format_exc()}")
        return {
            "success": False,
            "error": f"Error generando datos: {str(e)}"
        }