"""Clean citations from data"""
import json
import re
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from app.core.config import settings


def clean_citations(data_path: str = None):
    """Remove citation patterns from output field"""
    data_path = data_path or settings.DATA_PATH
    
    print(f"Loading data from {data_path}...")
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Regular expression to match [cite: numbers] pattern including ranges
    citation_pattern = r'\[cite:\s*[\d,\s\-]+\]'
    
    print("Removing citations...")
    for item in data:
        if 'output' in item:
            # Remove all citation patterns
            item['output'] = re.sub(citation_pattern, '', item['output'])
            # Clean up spaces
            item['output'] = re.sub(r'\s+', ' ', item['output'])
            item['output'] = re.sub(r'\s+([.,;:!?])', r'\1', item['output'])
            item['output'] = item['output'].strip()
    
    print(f"Saving cleaned data to {data_path}...")
    with open(data_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print("✅ Citations removed successfully!")


if __name__ == "__main__":
    clean_citations()
