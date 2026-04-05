"""
test_sandbox.py
────────────────────────────────────────────────────────────────────────────
Dedicated script to test the E2B Cloud Sandbox execution.

This script directly tests the `SandboxExecutor` to verify:
1. Your E2B_API_KEY is working.
2. The sandbox boots successfully.
3. Dependencies are installed.
4. Local files (core, tools) are uploaded successfully.
5. The chart renders remotely and is downloaded correctly.

Run via: python test_sandbox.py
"""

import os
import sys
import base64
import time
from dotenv import load_dotenv

def main():
    # Load environment variables
    load_dotenv()
    
    api_key = os.getenv("E2B_API_KEY")
    if not api_key:
        print("[ERROR] E2B_API_KEY is missing from your .env file!")
        print("Please add it before running the sandbox test.")
        sys.exit(1)
        
    print(f"[OK] Found E2B_API_KEY (starts with {api_key[:8]}...)")
    print("[INIT] Initializing SandboxExecutor...\n")
    
    try:
        from sandbox.executor import SandboxExecutor
    except ImportError:
        print("[ERROR] Could not import e2b. Please run: pip install 'e2b>=0.17'")
        sys.exit(1)

    # We use verbose=True so we can see the exact steps happening in the background!
    try:
        with SandboxExecutor(verbose=True) as executor:
            print("\n[Step 1 & 2 & 3] Booting sandbox, installing deps, and uploading local project files...")
            
            # Simple dataset
            data = {
                "x_col": ["Cloud", "Local", "Edge"],
                "y_col": [85, 45, 60],
                "x_col_name": "Infrastructure",
                "y_col_name": "Performance Score"
            }
            
            print("\n[Step 4] Requesting bar_chart render inside E2B Cloud...")
            t0 = time.time()
            
            # Process the tool in the cloud
            b64_string = executor.run_tool(
                tool_name="bar_chart",
                data=data,
                title="E2B Sandbox Render Test",
                options={"color": "#FF5733"}
            )
            
            elapsed = time.time() - t0
            print(f"\n[SUCCESS] Chart received in {elapsed:.2f} seconds.")
            
            # Save the file
            out_dir = "test_output"
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, "sandbox_test.png")
            
            with open(out_path, "wb") as f:
                f.write(base64.b64decode(b64_string))
                
            print(f"[IMAGE] Chart saved successfully to: {out_path}")
            
    except Exception as e:
        print(f"\n[FATAL ERROR] inside Sandbox Execution: {e}")

if __name__ == "__main__":
    main()
