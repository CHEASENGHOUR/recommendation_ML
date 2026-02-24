import pandas as pd
import numpy as np

def load_laptop_data(file_path: str) -> pd.DataFrame:
    """Load and clean laptop dataset"""
    df = pd.read_csv(file_path)
    
    # Clean column names
    df.columns = df.columns.str.strip().str.lower()
    
    # Create unique laptop_id
    df['laptop_id'] = range(len(df))
    
    # Categorize GPU
    df['gpu_type'] = df['gpu'].apply(categorize_gpu)
    df['cpu_tier'] = df['cpu'].apply(categorize_cpu)
    df['usage_type'] = df.apply(determine_usage_type, axis=1)
    
    return df

def categorize_gpu(gpu: str) -> str:
    gpu_lower = str(gpu).lower()
    if any(x in gpu_lower for x in ['rtx 4080', 'rtx 4090', 'rtx 5080']):
        return 'high_end'
    elif any(x in gpu_lower for x in ['rtx 3050', 'rtx 3060', 'rtx 4050', 'rtx 4060', 'rtx 4070']):
        return 'mid_range'
    elif any(x in gpu_lower for x in ['gtx', 'rx 6500', 'rx 6600']):
        return 'entry_gaming'
    elif 'integrated' in gpu_lower:
        return 'integrated'
    return 'other'

def categorize_cpu(cpu: str) -> str:
    cpu_lower = str(cpu).lower()
    if any(x in cpu_lower for x in ['i9', 'ryzen 9', 'core ultra 9']):
        return 'premium'
    elif any(x in cpu_lower for x in ['i7', 'ryzen 7', 'core ultra 7']):
        return 'high'
    elif any(x in cpu_lower for x in ['i5', 'ryzen 5', 'core ultra 5']):
        return 'mid'
    elif any(x in cpu_lower for x in ['i3', 'ryzen 3']):
        return 'budget'
    return 'entry'

def determine_usage_type(row) -> str:
    if row.get('gpu_vram', 0) > 0:
        return 'gaming'
    elif row['price'] > 80000:
        return 'professional'
    elif row['ram_capacity'] >= 16 and row['cpu_tier'] in ['high', 'premium']:
        return 'productivity'
    else:
        return 'everyday'