from mcp.server import FastMCP
import os

mcp = FastMCP("patch manager")


@mcp.tool()
def apply_patch(file_path: str, new_code: str) -> dict:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            old_code = f.read()

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(new_code)

        return {
            "status": "success",
            "file_path": file_path,
            "old_code": old_code,
            "new_code": new_code
        }
    
    except Exception as e:
        return {"error": f"Failed to patch file: {str(e)}"}

@mcp.tool()
def create_audit_log(file_path: str, summary: str) -> dict:
    name, _ = os.path.splitext(os.path.basename(file_path))
    report_filename = f"audit_report_{name}.md"
    
    try:
        content = f"# Security Audit Report\n\n**Target File:** `{file_path}`\n\n## Summary\n{summary}\n"
        
        with open(report_filename, "w", encoding="utf-8") as f:
            f.write(content)
            
        return {
            "success": True,
            "report_path": report_filename
        }
    except Exception as e:
        return {"error": f"Failed to create audit log: {str(e)}"}


if __name__ == "__main__":
    mcp.run()