import subprocess
import os
import threading

def run_backend_and_wait():
    print("[INFO] Starting backend and waiting for server startup...")

    backend_process = subprocess.Popen(
        ["python", "-m", "src.main"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        shell=True,
        bufsize=1
    )

    def monitor_backend_output():
        for line in backend_process.stdout:
            print("[BACKEND]", line.strip())
            if "Application startup complete." in line:
                print("[INFO] Backend is ready.")
                run_frontend()
                break

    threading.Thread(target=monitor_backend_output, daemon=True).start()
    return backend_process

def run_frontend():
    print("[INFO] Starting frontend...")
    frontend_path = os.path.join(os.getcwd(), "frontend")
    subprocess.Popen(["npm", "start", "dev"], cwd=frontend_path, shell=True)

if __name__ == "__main__":
    try:
        backend_process = run_backend_and_wait()
        print("[INFO] Press Ctrl+C to stop both processes.")
        backend_process.wait()
    except KeyboardInterrupt:
        print("\n[INFO] Shutting down...")
        backend_process.terminate()
