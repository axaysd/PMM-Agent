#!/usr/bin/env python3
"""
Startup script for PMM Assistant
"""

import os
import sys
from pathlib import Path

def check_environment():
    """Check if environment is properly set up"""
    print("🔍 Checking environment...")
    
    # Check if .env file exists
    env_file = Path(".env")
    if not env_file.exists():
        print("⚠️  No .env file found. Creating template...")
        with open(".env", "w") as f:
            f.write("OPENAI_API_KEY=your_openai_api_key_here\n")
        print("📝 Please edit .env file and add your OpenAI API key")
        return False
    
    # Check if API key is set
    from config import Config
    if not Config.OPENAI_API_KEY or Config.OPENAI_API_KEY == "your_openai_api_key_here":
        print("❌ OpenAI API key not configured")
        print("Please edit .env file and add your OpenAI API key")
        return False
    
    print("✅ Environment check passed")
    return True

def main():
    """Main startup function"""
    print("🚀 Starting PMM Assistant...")
    print("=" * 50)
    
    if not check_environment():
        print("\n❌ Environment setup incomplete. Please fix the issues above.")
        sys.exit(1)
    
    print("\n🎯 Starting FastAPI server...")
    print("📱 Open your browser to: http://localhost:8000")
    print("🛑 Press Ctrl+C to stop the server")
    print("=" * 50)
    
    # Import and run the main application
    try:
        from main import app
        import uvicorn
        from config import Config
        
        uvicorn.run(
            app, 
            host=Config.HOST, 
            port=Config.PORT,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n👋 Shutting down PMM Assistant...")
    except Exception as e:
        print(f"\n❌ Error starting application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
