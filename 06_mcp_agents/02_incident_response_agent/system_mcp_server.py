
from mcp.server.fastmcp import FastMCP
import os
import psutil
mcp = FastMCP('system monitor')

@mcp.tool()
def list_directory_logs(folder_path: str)->list[dict]:
    if not os.path.exists(folder_path):
        return []

    log_files = []
    for f in os.listdir(folder_path):
        if f.endswith(('.log', '.txt')):
            full_path = os.path.join(folder_path, f)
            size_bytes = os.path.getsize(full_path)

            log_files.append({'filename':full_path,'size in bytes':size_bytes})
    return log_files

@mcp.tool()
def read_system_metrics()->dict:

    cpu_percent = psutil.cpu_percent(interval=0.1)
    

    memory = psutil.virtual_memory()
    
    disk = psutil.disk_usage(os.environ.get('SystemDrive', 'C:\\') + os.sep)

    return{
        "cpu": {
            "usage_percent": cpu_percent,
            "core_count": psutil.cpu_count(logical=True)
        },
        "memory": {
            "total_gb": round(memory.total / (1024**3), 2),
            "used_gb": round(memory.used / (1024**3), 2),
            "percent_used": memory.percent
        },
        "disk": {
            "total_gb": round(disk.total / (1024**3), 2),
            "free_gb": round(disk.free / (1024**3), 2),
            "percent_used": disk.percent
        }
    }

if __name__ == "__main__":
    mcp.run()
