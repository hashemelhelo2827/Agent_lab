from mcp.server.fastmcp import FastMCP
import sqlite3

mcp = FastMCP('incident db')

conn = sqlite3.connect("incidents.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
    CREATE TABLE IF NOT EXISTS incidents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        severity TEXT,
        status TEXT,
        summary TEXT
    )
""")

cursor.execute("SELECT COUNT(*) FROM incidents")
if cursor.fetchone()[0] == 0:
    cursor.executemany(
        "INSERT INTO incidents (severity, status, summary) VALUES (?, ?, ?)",
        [
            ("high", "open", "High CPU load on production server"),
            ("medium", "open", "Database connection timeouts"),
            ("low", "resolved", "SSL certificate renewal complete"),
        ],
    )
    conn.commit()

@mcp.tool()
def query_incidents(status: str)->list[dict]:
    cursor.execute("SELECT id, severity, status, summary FROM incidents WHERE status = ?", (status,))
    rows=cursor.fetchall()

    return[{
        'id':row[0],
        'severity':row[1],
        'status':row[2],
        'summary':row[3],
    } for row in rows]

@mcp.tool()
def log_incident_resolution(incident_id: int, summary: str)->dict:
    cursor.execute("SELECT id, severity, status, summary FROM incidents WHERE id = ?", (incident_id,))
    row=cursor.fetchone()
    if not row:
        return {"error": f"Incident with ID {incident_id} not found."}

    
    inc_id, severity, old_status, old_summary = row
    updated_summary = f"{old_summary}\n[Resolution]: {summary}"
    new_status = "RESOLVED"
    cursor.execute(
        "UPDATE incidents SET status = ?, summary = ? WHERE id = ?",
        (new_status, updated_summary, incident_id)
    )
    
    
    conn.commit()
    
    return {
        "id": inc_id,
        "severity": severity,
        "status": new_status,
        "summary": updated_summary
    }

if __name__ == "__main__":
    mcp.run()

    
    
