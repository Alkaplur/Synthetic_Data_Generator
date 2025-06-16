"""
Versión simplificada de Pure Synthetic tools
"""

from langchain_ollama import OllamaLLM
import pandas as pd
import os

def clean_csv_response(response: str) -> str:
    """
    Limpia la respuesta del LLM para obtener CSV puro
    """
    # Remover comillas y símbolos de concatenación
    cleaned = response.replace('"', '').replace(' + ', '').replace('\\n', '\n')
    
    # Remover líneas que no parecen CSV
    lines = cleaned.split('\n')
    csv_lines = []
    
    for line in lines:
        line = line.strip()
        if line and ',' in line and not line.startswith('Sure') and not line.startswith('Here'):
            csv_lines.append(line)
    
    return '\n'.join(csv_lines)

def generate_simple_data(data_type: str, num_rows: int = 10):
    """
    Genera datos sintéticos usando Nemotron directamente
    """
    print(f"🎯 Generando {num_rows} filas de {data_type}...")
    
    # Crear LLM
    llm = OllamaLLM(
        model="nemotron-mini",
        base_url="http://localhost:11434",
        temperature=0.7,
    )
    
    # Prompt mejorado
    prompt = f"""You are a data generator. Generate exactly {num_rows} rows of {data_type} data.

Format: CSV with headers
Requirements:
- First line must be column headers
- Each following line is one data row
- Use realistic data
- No explanations, just the CSV data

Example format:
id,name,email,age
1,John Smith,john@email.com,25
2,Mary Johnson,mary@email.com,30

Now generate {num_rows} rows of {data_type} data:"""
    
    # Generar
    response = llm.invoke(prompt)
    print(f"✅ Respuesta generada: {len(response)} caracteres")
    
    # Limpiar respuesta
    clean_csv = clean_csv_response(response)
    print(f"🧹 CSV limpio: {len(clean_csv)} caracteres")
    
    # Guardar como archivo
    filename = f"{data_type}_synthetic_{num_rows}rows.csv"
    with open(filename, 'w') as f:
        f.write(clean_csv)
    
    print(f"📁 Archivo guardado: {filename}")
    
    # Mostrar preview
    lines = clean_csv.split('\n')
    print(f"📊 Preview (primeras 3 líneas):")
    for i, line in enumerate(lines[:3]):
        print(f"   {line}")
    
    return filename