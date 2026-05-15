import subprocess
import sys
import os
import signal
import time

def main():
    print("Starting MedStep...")
    print("=" * 40)
    
    # Start API
    api = subprocess.Popen(
        [sys.executable, "backend/api.py"],
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
    )
    print("✅ API server starting on http://localhost:8000")
    
    # Small delay to let API load models first
    time.sleep(2)
    
    # Start frontend
    frontend = subprocess.Popen(
        [sys.executable, "frontend_v2/main.py"],
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
    )
    print("✅ Frontend starting on http://localhost:8501")
    print("=" * 40)
    print("MedStep is running!")
    print("Open: http://localhost:8501")
    print("Press Ctrl+C to stop")
    print("=" * 40)

    try:
        api.wait()
    except KeyboardInterrupt:
        print("\nShutting down MedStep...")
        api.terminate()
        frontend.terminate()
        print("Stopped.")

if __name__ == "__main__":
    main()