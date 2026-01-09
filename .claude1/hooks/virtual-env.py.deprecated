import os
import sys
import subprocess

def ensure_venv():
    # Check if virtual environment is active
    if not hasattr(sys, 'real_prefix') and not getattr(sys, 'base_prefix', sys.prefix) != sys.prefix:
        # Look for venv in common locations
        venv_paths = [
            os.path.join(os.getcwd(), 'venv'),
            os.path.join(os.getcwd(), '.venv'),
            os.path.join(os.getcwd(), 'env')
        ]
        
        for venv_path in venv_paths:
            if os.path.exists(os.path.join(venv_path, 'Scripts', 'activate.bat')):  # Windows
                subprocess.run(f'"{os.path.join(venv_path, "Scripts", "activate.bat")}"', shell=True)
                os.environ['VIRTUAL_ENV'] = venv_path
                sys.path.append(os.path.join(venv_path, 'Scripts'))
                print(f"Activated virtual environment: {venv_path}")
                return
            elif os.path.exists(os.path.join(venv_path, 'bin', 'activate')):  # Unix/Linux/Mac
                subprocess.run(f'source {os.path.join(venv_path, "bin", "activate")}', shell=True)
                os.environ['VIRTUAL_ENV'] = venv_path
                sys.path.append(os.path.join(venv_path, 'bin'))
                print(f"Activated virtual environment: {venv_path}")
                return
                
        raise EnvironmentError("No virtual environment found in common locations.")

if __name__ == "__main__":
    ensure_venv()