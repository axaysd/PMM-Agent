#!/usr/bin/env python3
"""
Test script to verify PMM Assistant setup
"""

import sys
import os

def test_imports():
    """Test if all required packages can be imported"""
    try:
        import fastapi
        print("✅ FastAPI imported successfully")
    except ImportError as e:
        print(f"❌ FastAPI import failed: {e}")
        return False
    
    try:
        import langgraph
        print("✅ LangGraph imported successfully")
    except ImportError as e:
        print(f"❌ LangGraph import failed: {e}")
        return False
    
    try:
        import langchain_openai
        print("✅ LangChain OpenAI imported successfully")
    except ImportError as e:
        print(f"❌ LangChain OpenAI import failed: {e}")
        return False
    
    try:
        import pandas
        print("✅ Pandas imported successfully")
    except ImportError as e:
        print(f"❌ Pandas import failed: {e}")
        return False
    
    return True

def test_config():
    """Test configuration setup"""
    try:
        from config import Config
        print("✅ Config module imported successfully")
        
        # Check if API key is set
        if Config.OPENAI_API_KEY:
            print("✅ OpenAI API key is configured")
        else:
            print("⚠️  OpenAI API key not set - set OPENAI_API_KEY environment variable")
        
        return True
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False

def test_workflow():
    """Test LangGraph workflow initialization"""
    try:
        from langgraph_workflow import PositioningWorkflow
        
        # Only test if API key is available
        from config import Config
        if Config.OPENAI_API_KEY:
            workflow = PositioningWorkflow()
            print("✅ LangGraph workflow initialized successfully")
        else:
            print("⚠️  Skipping workflow test - no API key")
        
        return True
    except Exception as e:
        print(f"❌ Workflow test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing PMM Assistant Setup")
    print("=" * 40)
    
    tests = [
        ("Package Imports", test_imports),
        ("Configuration", test_config),
        ("LangGraph Workflow", test_workflow),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Testing {test_name}...")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} test failed")
    
    print("\n" + "=" * 40)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! PMM Assistant is ready to run.")
        print("\nTo start the application:")
        print("  python main.py")
        print("\nThen open your browser to:")
        print("  http://localhost:8000")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
