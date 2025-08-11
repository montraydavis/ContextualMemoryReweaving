#!/usr/bin/env python3
"""
Simple script to load environment variables from .env file.
"""

import os
from pathlib import Path

def load_dotenv():
    """Load environment variables from .env file if it exists."""
    # Get the project root directory (parent of python directory)
    project_root = Path(__file__).parent.parent
    env_file = project_root / ".env"
    
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Remove quotes if present
                    if (value.startswith('"') and value.endswith('"')) or \
                       (value.startswith("'") and value.endswith("'")):
                        value = value[1:-1]
                    
                    os.environ[key] = value

if __name__ == "__main__":
    load_dotenv()
