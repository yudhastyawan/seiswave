import sys
import os
import subprocess

def main():
    """CLI entry point for seiswave-web"""
    script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "web_app.py"))
    print(f"Starting seiswave Web UI from: {script_path}")
    
    # Run streamlit via subprocess. Replace current python process if POSIX.
    args = [sys.executable, "-m", "streamlit", "run", script_path] + sys.argv[1:]
    
    if os.name == 'posix':
        os.execvp(args[0], args)
    else:
        sys.exit(subprocess.call(args))

if __name__ == "__main__":
    main()
