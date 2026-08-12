import sys 
from mcp.server.fastmcp import FastMCP

mcp=FastMCP("Engineering-Tools")

@mcp.tool()
def calculate_matrix_trace(matrix:list[list[float]])->float:
    return sum(matrix[i][i] for i in range(len(matrix)))

@mcp.tool()
def read_system_status(service_name : str)->str:
    status={
        "database": "Operational (Lat 2ms)",
        "vectorstore": "Degraded (High memory usage)",
        "llm_gateway": "Operational"

    }
    return status.get(service_name.lower(), "Unknown Service")

if __name__ == "__main__":
    # If passed a "test" argument, run functions locally to verify logic
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        print("🚀 Testing MCP Tool Functions Locally...\n")
        
        matrix_result = calculate_matrix_trace([[1.0, 2.0], [3.0, 4.0]])
        print(f"✅ Trace Test Result: {matrix_result}")
        
        status_result = read_system_status("vectorstore")
        print(f"✅ Status Test Result: {status_result}\n")
        print("MCP Server tools are working! Run without 'test' to start stdio server.")
    else:
        # Standard MCP Server entry point (communicates over stdio)
        print("⚡ Starting MCP Server via stdio...", file=sys.stderr)
        mcp.run(transport="stdio")