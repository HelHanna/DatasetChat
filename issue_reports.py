import os
import logging
import uuid
import sqlite3
import sys
import pandas as pd
from datetime import datetime
from typing import Set, List, Dict, Optional, Tuple, Any, Callable

# ============================================================
# LOGGING SETUP
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stderr),
        logging.FileHandler("tool_calls.log")
    ]
)

logger = logging.getLogger(__name__)
# ── Suppress HTTP Request Logs ──
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)


# ============================================================
# ISSUE REPORT MANAGER CLASS
# ============================================================
os.makedirs("/data", exist_ok=True)
db_path = "/data/issue_reports.db"

class IssueReportManager:
    def __init__(self, db_path=db_path):
        self.db_path = db_path
        self.init_db()

    def init_db(self):
        """Initialize the issue reports database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS issue_reports (
                    id TEXT PRIMARY KEY,
                    session_id TEXT,
                    timestamp TEXT,
                    category TEXT,
                    title TEXT,
                    description TEXT,
                    severity TEXT,
                    screenshot_base64 TEXT,
                    browser_info TEXT,
                    status TEXT DEFAULT 'open',
                    admin_notes TEXT DEFAULT ''
                )
            """)
            conn.commit()
        logger.info("✅ Issue reports table initialized")

    def report_issue(self, session_id: str, category: str, title: str, 
                     description: str, severity: str, screenshot_b64: str = None, 
                     browser_info: str = None) -> str:
        """Create a new issue report"""
        report_id = str(uuid.uuid4())
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO issue_reports 
                   (id, session_id, timestamp, category, title, description, severity, screenshot_base64, browser_info) 
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (report_id, session_id, datetime.now().isoformat(), category, title, 
                 description, severity, screenshot_b64, browser_info)
            )
            conn.commit()
        logger.info(f"📋 Issue report created: {report_id} - {title}")
        return report_id

    def get_all_reports(self) -> List[Dict]:
        """Get all issue reports for admin (summary view)"""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("""
                SELECT id, session_id, timestamp, category, title, severity, status, description
                FROM issue_reports
                ORDER BY timestamp DESC
            """).fetchall()
        
        return [
            {
                "ID": r[0][:12] + "...",
                "Session": r[1][:8] + "..." if r[1] else "N/A",
                "Timestamp": r[2],
                "Category": r[3],
                "Title": r[4],
                "Severity": r[5],
                "Status": r[6],
                "Preview": r[7][:80] + "..." if len(r[7]) > 80 else r[7]
            }
            for r in rows
        ]

    def get_report_details(self, report_id: str) -> Dict:
        """Get full details of a specific report"""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("""
                SELECT id, session_id, timestamp, category, title, description, 
                       severity, screenshot_base64, browser_info, status, admin_notes
                FROM issue_reports
                WHERE id = ?
            """, (report_id,)).fetchone()
        
        if not row:
            return None
        
        return {
            "id": row[0],
            "session_id": row[1],
            "timestamp": row[2],
            "category": row[3],
            "title": row[4],
            "description": row[5],
            "severity": row[6],
            "has_screenshot": bool(row[7]),
            "browser_info": row[8],
            "status": row[9],
            "admin_notes": row[10]
        }

    def update_report_status(self, report_id: str, status: str, admin_notes: str = ""):
        """Update report status and add admin notes"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE issue_reports SET status = ?, admin_notes = ? WHERE id = ?",
                (status, admin_notes, report_id)
            )
            conn.commit()
        logger.info(f"🔧 Updated report {report_id[:12]} status to {status}")

    def download_reports(self) -> str:
        """Export all reports to CSV"""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("""
                SELECT timestamp, session_id, category, title, severity, status, description
                FROM issue_reports
                ORDER BY timestamp DESC
            """).fetchall()
        
        df = pd.DataFrame(rows, columns=[
            "Timestamp", "Session ID", "Category", "Title", "Severity", "Status", "Description"
        ])
        
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f, index=False)
            temp_path = f.name
        
        logger.info(f"📥 Reports exported to CSV")
        return temp_path

    def get_statistics(self) -> Dict:
        """Get issue report statistics"""
        with sqlite3.connect(self.db_path) as conn:
            total = conn.execute("SELECT COUNT(*) FROM issue_reports").fetchone()[0]
            by_status = conn.execute(
                "SELECT status, COUNT(*) FROM issue_reports GROUP BY status"
            ).fetchall()
            by_severity = conn.execute(
                "SELECT severity, COUNT(*) FROM issue_reports GROUP BY severity"
            ).fetchall()
            by_category = conn.execute(
                "SELECT category, COUNT(*) FROM issue_reports GROUP BY category"
            ).fetchall()
        
        return {
            "total": total,
            "by_status": dict(by_status) if by_status else {},
            "by_severity": dict(by_severity) if by_severity else {},
            "by_category": dict(by_category) if by_category else {}
        }

issue_report_manager = IssueReportManager()



# ============================================================
# ISSUE REPORT HANDLER FUNCTIONS
# ============================================================

def submit_issue_report(category, title, description, severity, session_id):
    """Handle issue report submission"""
    if not title.strip() or not description.strip() or not category:
        return "❌ Please fill in all required fields", False

    browser_info = "Captured via Gradio Interface"

    try:
        report_id = issue_report_manager.report_issue(
            session_id=session_id,
            category=category,
            title=title,
            description=description,
            severity=severity,
            browser_info=browser_info
        )
        return f"✅ Report submitted! (ID: {report_id[:12]}...)", True
    except Exception as e:
        logger.error(f"Issue report submission failed: {e}")
        return f"❌ Error submitting report: {str(e)}", False


def get_issue_reports_admin(admin_key):
#     """Get all issue reports for admin view"""
#     if admin_key != os.getenv("ADMIN_KEY"):
#         return "Unauthorized"
    reports = issue_report_manager.get_all_reports()
    if not reports:
        return pd.DataFrame()
    return pd.DataFrame(reports)


def get_issue_stats(admin_key):
    # """Get issue report statistics as HTML"""
    # if admin_key != os.getenv("ADMIN_KEY"):
    #     return '<div style="color: red; padding: 20px;">Unauthorized</div>'

    try:
        stats = issue_report_manager.get_statistics()

        severity_colors = {
            "critical": "#ef4444",
            "high": "#f97316",
            "medium": "#eab308",
            "low": "#6366f1"
        }

        severity_html = ""
        for severity in ["critical", "high", "medium", "low"]:
            count = stats['by_severity'].get(severity, 0)
            color = severity_colors.get(severity, "#6b7280")
            severity_html += (
                f'<div style="background: {color}22; border: 2px solid {color}; '
                f'padding: 8px 12px; border-radius: 8px; color: white; font-weight: 600; '
                f'display: inline-block; margin-right: 8px; margin-bottom: 8px;">'
                f'{severity.title()}: {count}</div>'
            )

        html = f"""
        <div style="padding: 20px; background: var(--bg-secondary); border-radius: 12px; 
                    color: var(--text-primary); font-family: 'Poppins', sans-serif;">
            <h3 style="margin-top: 0; color: var(--accent-cyan);">📊 Issue Report Statistics</h3>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 16px; margin: 20px 0;">
                <div style="background: var(--bg-dark); padding: 16px; border-radius: 8px; border-left: 4px solid var(--accent-cyan);">
                    <div style="font-size: 12px; text-transform: uppercase; color: var(--text-muted); margin-bottom: 8px; font-weight: 600;">Total Reports</div>
                    <div style="font-size: 28px; font-weight: 700; color: var(--accent-cyan);">{stats['total']}</div>
                </div>
                <div style="background: var(--bg-dark); padding: 16px; border-radius: 8px; border-left: 4px solid #f59e0b;">
                    <div style="font-size: 12px; text-transform: uppercase; color: var(--text-muted); margin-bottom: 8px; font-weight: 600;">Open</div>
                    <div style="font-size: 28px; font-weight: 700; color: #fbbf24;">{stats['by_status'].get('open', 0)}</div>
                </div>
                <div style="background: var(--bg-dark); padding: 16px; border-radius: 8px; border-left: 4px solid #10b981;">
                    <div style="font-size: 12px; text-transform: uppercase; color: var(--text-muted); margin-bottom: 8px; font-weight: 600;">Resolved</div>
                    <div style="font-size: 28px; font-weight: 700; color: #6ee7b7;">{stats['by_status'].get('resolved', 0)}</div>
                </div>
            </div>
            <div style="margin-top: 20px;">
                <h4 style="color: var(--accent-cyan); margin-bottom: 12px;">By Severity:</h4>
                <div style="display: flex; gap: 12px; flex-wrap: wrap;">
                    {severity_html}
                </div>
            </div>
        </div>
        """
        return html
    except Exception as e:
        logger.error(f"Error getting issue stats: {e}")
        return f'<div style="color: red; padding: 20px;">Error loading statistics: {str(e)}</div>'


def download_issue_reports(admin_key):
    """Download all issue reports as CSV"""
    if admin_key != os.getenv("ADMIN_KEY"):
        return '<div style="color: red; padding: 20px;">Unauthorized</div>'

    try:
        return issue_report_manager.download_reports()
    except Exception as e:
        logger.error(f"Error downloading reports: {e}")
        return None
