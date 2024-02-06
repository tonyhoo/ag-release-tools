import subprocess
import sys

def run_pip_test():
    run_pip_check()
    run_safety_check()

def run_pip_check():
    result = subprocess.run(["pip", "check"], capture_output=True, text=True)
    if result.returncode == 0:
        print("pip check passed.")
        print(result.stdout)
    else:
        print("pip check failed.")
        print(result.stdout)

def run_safety_check():
    # Ensure safety is installed (output is not captured here)
    subprocess.run(["pip", "install", "-U", "safety"], check=True)
    
    # Run safety check and capture the output
    result = subprocess.run(["safety", "check", "--full-report"], capture_output=True, text=True)
    if result.returncode == 0:
        print("safety check passed.")
        print(result.stdout)
    else:
        print("safety check found issues.")
        print(result.stdout)