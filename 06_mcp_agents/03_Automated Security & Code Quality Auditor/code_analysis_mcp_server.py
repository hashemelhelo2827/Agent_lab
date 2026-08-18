from mcp.server import FastMCP
import ast
import re 

mcp = FastMCP("code_analysis")

@mcp.tool()
def analyze_syntax(code_string: str) -> dict:
    defcount = 0
    importcount = 0
    classcount = 0
    try:
        tree = ast.parse(code_string)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                defcount += 1 
            elif isinstance(node, ast.ClassDef):
                classcount += 1 
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                importcount += 1 

        return {
            "valid_syntax": True,
            "defcount": defcount,
            "importcount": importcount,
            "classcount": classcount
        }
    
    except SyntaxError as e:
        return {
            "valid_syntax": False,
            "error": e.msg,
            "line": e.lineno,
            "column": e.offset
        }

@mcp.tool()
def run_linter(file_path: str) -> dict:
    issue = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            code = f.read()
    except Exception as e:
        return {"error": f"Failed to read file: {str(e)}"}

    secret_key = r'(?i)(api_key|secret|password|token)\s*=\s*["\'][^"\']+["\']'
    for line_num, line in enumerate(code.splitlines(), start=1):
        if re.search(secret_key, line):
            issue.append({
                "type": "secret key hard coded",
                "line": line_num,
                "message": "Possible hardcoded credential or secret key detected."
            })
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            # FIX: Use node.func.id instead of node.name
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in {"eval", "exec"}:
                    issue.append({
                        "type": "DANGEROUS_EVAL",
                        "line": node.lineno,
                        "message": f"Use of dangerous function '{node.func.id}()' detected."
                    })

            if isinstance(node, ast.ExceptHandler) and node.type is None:
                issue.append({
                    "type": "BARE_EXCEPT",
                    "line": node.lineno,
                    "message": "Bare except clause used; catches all exceptions unconditionally."
                })
    except SyntaxError as e:
        return {"error": f"Syntax error prevents linting: {e.msg} at line {e.lineno}"}

    return {
        "file_path": file_path,
        "issue_count": len(issue),
        "issues": issue
    }


if __name__ == "__main__":
    mcp.run()