"""
Pure Synthetic Tools - Versión dinámica sin templates fijos
"""

import os
import pandas as pd
import openai
from typing import Dict, Any
from pathlib import Path
from agents import function_tool, RunContextWrapper
from sdk_tools_and_context import SyntheticDataContext

# Directorio base del proyecto
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "Synthetic data generated"

# Configurar cliente NVIDIA
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
        return f"Error: {str(e)}"

def clean_csv_response(response: str) -> str:
    """Limpiar respuesta del LLM para obtener CSV puro"""
    lines = response.split('\n')
    csv_lines = []
    
    for line in lines:
        line = line.strip()
        
        # Quitar caracteres problemáticos
        line = line.replace('\\', '').replace('```csv', '').replace('```', '')
        
        # Es línea de headers o datos válidos?
        if (line and 
            ',' in line and
            not line.startswith('Sure') and
            not line.startswith('Here') and
            not line.startswith('I can') and
            not line.startswith('Generate') and
            not '[' in line and
            'example' not in line.lower()):
            csv_lines.append(line)
    
    return '\n'.join(csv_lines)

@function_tool
def generate_synthetic_data_dynamic(
    wrapper: RunContextWrapper[SyntheticDataContext],
    description: str,
    num_rows: int,
    country: str = "Spain"
) -> Dict[str, Any]:
    """
    Genera datos sintéticos dinámicamente basándose en la descripción del usuario.
    
    Args:
        description: Descripción libre de qué datos necesita (ej: "clientes de una tienda online")
        num_rows: Número de filas a generar
        country: País para localización de datos
        
    Returns:
        Información del archivo generado
    """
    try:
        context = wrapper.context
        
        # Validaciones
        if num_rows <= 0 or num_rows > 500:
            return {
                "success": False,
                "error": f"Número de filas debe estar entre 1 y 500, recibido: {num_rows}"
            }
        
        if len(description.strip()) < 5:
            return {
                "success": False,
                "error": "La descripción debe tener al menos 5 caracteres"
            }
        
        # PASO 1: Que Nemotron diseñe la estructura
        design_prompt = f"""You are a data structure expert. Based on this description: "{description}"
        
Design the optimal CSV structure for this type of data in {country}.

Requirements:
- Think about what columns would be most useful and realistic
- Include 6-12 relevant columns
- Use appropriate data types
- Consider the context of {country}

Output ONLY the CSV headers (first line), nothing else.
Example: id,name,email,age,city,registration_date"""

        headers_response = call_nvidia_nemotron(design_prompt, 500)
        headers = clean_csv_response(headers_response).strip()
        
        if not headers or ',' not in headers:
            return {
                "success": False,
                "error": f"No se pudo generar estructura válida. Respuesta: {headers_response[:100]}"
            }
        
        # PASO 2: Generar los datos usando la estructura diseñada
        data_prompt = f"""Generate exactly {num_rows} rows of realistic data for: {description}

Use this exact structure: {headers}

Requirements:
- Generate exactly {num_rows} data rows
- Make data realistic for {country}
- Use appropriate {country} names, cities, dates, etc.
- Start with the headers, then {num_rows} data rows
- Output ONLY CSV data, no explanations

Headers: {headers}
Data: (generate {num_rows} rows)"""

        response = call_nvidia_nemotron(data_prompt, 100000)
        
        # Limpiar respuesta
        clean_csv = clean_csv_response(response)
        
        if not clean_csv or len(clean_csv.strip()) < 10:
            clean_csv = response
        
        # Directorio de salida específico
        output_dir = str(OUTPUT_DIR)
        os.makedirs(output_dir, exist_ok=True)
        
        # Crear nombre de archivo descriptivo
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        # Limpiar descripción para nombre de archivo
        clean_desc = "".join(c for c in description if c.isalnum() or c in (' ', '-', '_')).rstrip()
        clean_desc = clean_desc.replace(' ', '_')[:30]  # Max 30 chars
        filename = f"{clean_desc}_synthetic_{num_rows}rows_{timestamp}.csv"
        full_path = os.path.join(output_dir, filename)
        
        # Guardar archivo
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(clean_csv)
        
        # Actualizar contexto
        context.generated_file_path = full_path
        context.generated_file_id = timestamp
        context.generated_rows = num_rows
        context.last_model_used = "nvidia/llama-3.1-nemotron-70b-instruct"
        
        # Extraer info del CSV generado
        lines = clean_csv.split('\n')
        actual_headers = lines[0] if lines else headers
        actual_rows = len(lines) - 1 if len(lines) > 1 else 0
        
        return {
            "success": True,
            "filename": filename,
            "rows_generated": actual_rows,
            "description": description,
            "headers_designed": actual_headers,
            "file_path": full_path,
            "country": country,
            "message": f"✅ Generados {actual_rows} registros de '{description}' usando Nemotron 70B"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Error generando datos: {str(e)}"
        }
