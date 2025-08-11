#!/usr/bin/env python3
"""
Simple test script to check if imports work correctly.
"""

import sys
import os

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(__file__))

try:
    from models.base_transformer import CMRTransformer
    print("✅ CMRTransformer imported successfully")
except ImportError as e:
    print(f"❌ Failed to import CMRTransformer: {e}")

try:
    from models.memory_buffer import LayeredMemoryBuffer
    print("✅ LayeredMemoryBuffer imported successfully")
except ImportError as e:
    print(f"❌ Failed to import LayeredMemoryBuffer: {e}")

try:
    from models.relevance_scorer import RelevanceScorer
    print("✅ RelevanceScorer imported successfully")
except ImportError as e:
    print(f"❌ Failed to import RelevanceScorer: {e}")

try:
    from models.reconstruction import LayeredStateReconstructor
    print("✅ LayeredStateReconstructor imported successfully")
except ImportError as e:
    print(f"❌ Failed to import LayeredStateReconstructor: {e}")

try:
    from models.advanced_retrieval import AdvancedMemoryRetriever
    print("✅ AdvancedMemoryRetriever imported successfully")
except ImportError as e:
    print(f"❌ Failed to import AdvancedMemoryRetriever: {e}")

try:
    from utils.hooks import HookManager
    print("✅ HookManager imported successfully")
except ImportError as e:
    print(f"❌ Failed to import HookManager: {e}")

print("\n🎯 All import tests completed!")
