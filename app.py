import gradio as gr
import os
from huggingface_hub import InferenceClient, HfApi
import json
import re
import requests
import io
import pandas as pd
from io import BytesIO
import logging
import sys
import time
import sqlite3
import uuid
from datetime import datetime
from dataclasses import dataclass, field
from typing import Set, List, Dict, Optional, Tuple, Any, Callable
from enum import Enum
from collections import defaultdict

from chatbot_agents import tag_matching_agent, tag_selection_agent, semantic_paper_search_agent, prefilter_judgement_agent, paper_relevance_judgement_agent, query_understanding_agent, full_context_relevance_judgement_agent, reranking_agent
from tools import tools
from issue_reports import get_issue_reports_admin, get_issue_stats, download_issue_reports, submit_issue_report

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
# MCP SERVER
# ============================================================

MCP_SERVER = "https://huggingface.co/mcp"

# ============================================================
# QUERY LOGGER
# ============================================================

os.makedirs("/data", exist_ok=True)
db_path = "/data/query_logs.db"

class QueryLogger:
    def __init__(self, db_path=db_path):
        self.db_path = db_path
        self.init_db()

    def init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS queries (
                    id TEXT PRIMARY KEY,
                    session_id TEXT,
                    timestamp TEXT,
                    user_query TEXT,
                    system_response TEXT
                )
            """)

    def log_query(self, session_id, query):
        qid = str(uuid.uuid4())
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO queries VALUES (?, ?, ?, ?, ?)",
                (qid, session_id, datetime.now().isoformat(), query, None)
            )
        return qid

    def update_query(self, qid, response):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE queries SET system_response=? WHERE id=?",
                (response, qid)
            )

query_logger = QueryLogger()


def get_all_queries(admin_key):
    if admin_key != os.getenv("ADMIN_KEY"):
        return "Unauthorized"

    conn = sqlite3.connect("/data/query_logs.db")
    rows = conn.execute("""
        SELECT timestamp, session_id, user_query, system_response
        FROM queries
        ORDER BY timestamp DESC
        LIMIT 100
    """).fetchall()
    conn.close()

    result = [
        [r[0], r[1], r[2], r[3]] for r in rows
    ]
    return result


def download_logs(admin_key):
    if admin_key != os.getenv("ADMIN_KEY"):
        return None
    
    conn = sqlite3.connect("/data/query_logs.db")
    rows = conn.execute("""
        SELECT timestamp, session_id, user_query, system_response
        FROM queries
        ORDER BY timestamp DESC
    """).fetchall()
    conn.close()
    
    df = pd.DataFrame(rows, columns=["Timestamp", "Session ID", "Query", "Response"])
    
    # Save to temporary file
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df.to_csv(f, index=False)
        temp_path = f.name
    
    return temp_path

# ============================================================
# SESSION MANAGER
# ============================================================

class SessionManager:
    def __init__(self):
        self.sessions = {}  

    def validate_token(self, token):
        if not token:
            return False
        try:
            list(HfApi(token=token).list_datasets(limit=1))
            return True
        except:
            return False

    def set_token(self, session_id, token):
        self.sessions[session_id] = token

    def get_token(self, session_id):
        return self.sessions.get(session_id)

session_manager = SessionManager()

def init_session():
    return str(uuid.uuid4())

# ============================================================
# LOG TREE TRACKER
# ============================================================

class LogTreeTracker:
    """
    Collects all log steps and renders them as a beautiful tree structure.
    """

    ICONS = {
        "search":   "🔍",
        "dataset":  "📊",
        "paper":    "📄",
        "model":    "🤖",
        "edge":     "🔗",
        "filter":   "🔽",
        "rank":     "🏆",
        "tag":      "🏷️",
        "fallback": "🔁",
        "done":     "✅",
        "warn":     "⚠️",
        "error":    "❌",
        "info":     "ℹ️",
    }

    def __init__(self):
        self.logs: List[Dict] = []
        self.nodes: Dict[str, Dict] = {}
        self.edges: List[Dict] = []
        self.tree_structure: List[Dict] = []

    def log(self, kind: str, text: str):
        """Add a log entry"""
        self.logs.append({"kind": kind, "text": text, "ts": time.time()})
        logger.info(f"[tracker] {kind}: {text}")
        self._rebuild_tree()

    def on_node(self, node_id: str, node_type: str, metadata: Dict = None):
        """Track a new knowledge graph node"""
        if node_id not in self.nodes:
            self.nodes[node_id] = {"type": node_type, "metadata": metadata or {}}
            icon = self.ICONS.get(node_type, "•")
            self.log(node_type, f"Added {node_type}: {node_id}")

    def on_edge(self, source: str, target: str, relation: str):
        """Track a new knowledge graph edge"""
        self.edges.append({"source": source, "target": target, "relation": relation})
        self.log("edge", f"{source} → [{relation}] → {target}")

    def _rebuild_tree(self):
        """Rebuild tree structure from logs"""
        self.tree_structure = []
        level_map = {}
        current_level = 1

        for idx, log in enumerate(self.logs):
            kind = log['kind']
            text = log['text']
            
            if kind not in level_map:
                level_map[kind] = current_level
                current_level += 1
            
            tree_node = {
                'id': idx,
                'kind': kind,
                'text': text,
                'level': level_map[kind],
                'icon': self.ICONS.get(kind, "•"),
            }
            self.tree_structure.append(tree_node)

    def counts(self):
        """Get counts by node type"""
        c = {"dataset": 0, "paper": 0, "model": 0}
        for n in self.nodes.values():
            t = n["type"]
            if t in c:
                c[t] += 1
        return c

    def render_tree_svg(self, theme: str = "dark") -> str:
        """Render logs as a scrollable, properly sized SVG tree"""
        if not self.tree_structure:
            return """<div style="text-align: center; padding: 40px; color: #999; font-size: 14px;">
                        🌳 Logs will appear here as you work…
                    </div>"""

        node_width = 220
        node_height = 60
        h_spacing = 260
        v_spacing = 110
        margin_top = 40
        margin_left = 40

        level_counts = defaultdict(int)
        for node in self.tree_structure:
            level_counts[node['level']] += 1

        max_nodes_in_row = max(level_counts.values(), default=1)
        max_level = max(n['level'] for n in self.tree_structure)

        svg_width = max(800, max_nodes_in_row * h_spacing + margin_left * 2)
        svg_height = max(500, max_level * v_spacing + 100)

        node_positions = {}
        level_x_counters = defaultdict(lambda: margin_left)

        for node in self.tree_structure:
            level = node['level']
            x = level_x_counters[level]
            y = margin_top + (level - 1) * v_spacing

            node_positions[node['id']] = {'x': x, 'y': y, 'level': level}
            level_x_counters[level] += h_spacing

        if theme == "light":
            bg_gradient = "linear-gradient(135deg, #e0f4ff 0%, #cff0ff 100%)"
            tree_path_stroke = "#0ea5e9"
            tree_path_opacity = "0.6"
        else:
            bg_gradient = "linear-gradient(135deg, #1a2438 0%, #151d2d 100%)"
            tree_path_stroke = "#00d9ff"
            tree_path_opacity = "0.5"

        svg = f'''
        <div style="overflow: auto; width: 100%; height: 100%;">
            <svg width="{svg_width}" height="{svg_height}" 
                xmlns="http://www.w3.org/2000/svg"
                style="background: {bg_gradient};
                        border-radius: 12px;
                        display: block;
                        transform-origin: 0 0;
                        transition: transform 0.2s ease;">
        '''

        for i, node in enumerate(self.tree_structure):
            if node['level'] > 1:
                for j in range(i - 1, -1, -1):
                    if self.tree_structure[j]['level'] == node['level'] - 1:
                        parent = node_positions[self.tree_structure[j]['id']]
                        child = node_positions[node['id']]

                        parent_x = parent['x'] + node_width / 2
                        parent_y = parent['y'] + node_height
                        child_x = child['x'] + node_width / 2
                        child_y = child['y']

                        mid_y = (parent_y + child_y) / 2
                        path = f"M {parent_x} {parent_y} Q {parent_x} {mid_y} {child_x} {child_y}"

                        svg += f'<path d="{path}" stroke="{tree_path_stroke}" stroke-width="2" fill="none" opacity="{tree_path_opacity}"/>'
                        break

        kind_colors = {
            'search': {'border': '#06b6d4', 'fill': '#ecf0ff'},
            'dataset': {'border': '#3b82f6', 'fill': '#eff6ff'},
            'paper': {'border': '#8b5cf6', 'fill': '#faf5ff'},
            'model': {'border': '#10b981', 'fill': '#f0fdf4'},
            'filter': {'border': '#f59e0b', 'fill': '#fffbf0'},
            'rank': {'border': '#ef4444', 'fill': '#fef2f2'},
            'done': {'border': '#10b981', 'fill': '#f0fdf4'},
            'error': {'border': '#dc2626', 'fill': '#fef2f2'},
            'edge': {'border': '#a78bfa', 'fill': '#faf5ff'},
            'tag': {'border': '#ec4899', 'fill': '#fff5f8'},
            'info': {'border': '#6366f1', 'fill': '#ecf0ff'},
            'warn': {'border': '#eab308', 'fill': '#fefce8'},
            'fallback': {'border': '#f59e0b', 'fill': '#fffbf0'},
        }

        for node in self.tree_structure:
            pos = node_positions[node['id']]
            colors = kind_colors.get(node['kind'], {'border': '#6b7280', 'fill': '#f9fafb'})

            x, y = pos['x'], pos['y']
            text = node['text']
            if len(text) > 28:
                text = text[:25] + '…'

            svg += f'''
            <g>
                <rect x="{x}" y="{y}" width="{node_width}" height="{node_height}"
                    rx="10" ry="10"
                    fill="{colors['fill']}" stroke="{colors['border']}" stroke-width="2"
                    style="filter: drop-shadow(0 3px 6px rgba(0,0,0,0.08))"/>
                
                <text x="{x + 10}" y="{y + 20}" font-size="13" fill="#6b7280" font-weight="600">
                    {node['icon']}
                </text>

                <text x="{x + 10}" y="{y + 40}" font-size="12" fill="#374151"
                    font-family="Arial, sans-serif">
                    {text}
                </text>
            </g>
            '''
        svg += """
            <script>
            setTimeout(() => {
                const panel = document.getElementById('tree-panel');
                if (panel) {
                    panel.scrollTop = panel.scrollHeight;
                }
            }, 50);
            </script>
            """
        svg += "</svg></div>"
        return svg


# ============================================================
# KNOWLEDGE GRAPH DATA STRUCTURES
# ============================================================

class NodeType(Enum):
    DATASET = "dataset"
    PAPER = "paper"
    MODEL = "model"

class RelationType(Enum):
    HAS_PAPER = "has_paper"
    HAS_DATASET = "has_dataset"
    HAS_MODEL = "has_model"
    SHARES_TAG = "shares_tag"

@dataclass
class KnowledgeNode:
    id: str
    type: NodeType
    metadata: Dict = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return isinstance(other, KnowledgeNode) and self.id == other.id

@dataclass
class KnowledgeEdge:
    source: KnowledgeNode
    target: KnowledgeNode
    relation: RelationType
    metadata: Dict = field(default_factory=dict)


# ============================================================
# MCP CLIENT
# ============================================================

class MCPPromptClient:
    def __init__(self, server_url, token):
        self.server_url = server_url
        self.token = token
        self.session_id = None

    def initialize_session(self):
        payload = {
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "python-client", "version": "1.0.0"}
            }
        }
        headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        try:
            r = requests.post(self.server_url, json=payload, headers=headers, timeout=10)
            if r.ok:
                self.session_id = r.headers.get("Mcp-Session-Id")
                logger.info(f"✅ MCP Session: {self.session_id}")
                return True
        except Exception as e:
            logger.error(f"MCP init error: {e}")
        return False

    def _headers(self):
        h = {"Content-Type": "application/json",
             "Accept": "application/json, text/event-stream",
             "Mcp-Session-Id": self.session_id}
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        return h

    def get_prompt(self, prompt_name: str, params: Dict) -> Optional[str]:
        if not self.session_id:
            self.initialize_session()
        if not self.session_id:
            return None
        payload = {"jsonrpc": "2.0", "id": int(time.time()*1000),
                   "method": "prompts/get",
                   "params": {"name": prompt_name, "arguments": params}}
        try:
            r = requests.post(self.server_url, json=payload, headers=self._headers(), timeout=15)
            if r.ok:
                result = r.json().get("result", {})
                messages = result.get("messages", [])
                if messages:
                    content = messages[0].get("content", {})
                    if isinstance(content, dict):
                        return content.get("text", "")
                    elif isinstance(content, str):
                        return content
        except Exception as e:
            logger.error(f"❌ Prompt error for {prompt_name}: {e}")
        return None

    def call_tool(self, tool_name: str, arguments: Dict) -> Optional[Dict]:
        if not self.session_id:
            self.initialize_session()
        if not self.session_id:
            return None
        payload = {"jsonrpc": "2.0", "id": int(time.time()*1000),
                   "method": "tools/call",
                   "params": {"name": tool_name, "arguments": arguments}}
        try:
            r = requests.post(self.server_url, json=payload, headers=self._headers(), timeout=30)
            if r.ok:
                return r.json().get("result", {})
            else:
                logger.error(f"❌ Tool call failed: {r.status_code}")
        except requests.exceptions.Timeout:
            logger.error(f"⏰ Tool call timeout for {tool_name}")
        except Exception as e:
            logger.error(f"❌ Tool error for {tool_name}: {e}", exc_info=True)
        return None

# ============================================================
# KNOWLEDGE GRAPH CLASS
# ============================================================

class KnowledgeGraph:
    def __init__(self, tracker: LogTreeTracker = None):
        self.nodes: Dict[str, KnowledgeNode] = {}
        self.edges: List[KnowledgeEdge] = []
        self.search_path: List[str] = []
        self.tracker = tracker

    def add_node(self, node: KnowledgeNode) -> KnowledgeNode:
        if node.id not in self.nodes:
            self.nodes[node.id] = node
            logger.info(f"📍 Added node: {node.type.value} - {node.id}")
            if self.tracker:
                self.tracker.on_node(node.id, node.type.value, node.metadata)
        return self.nodes[node.id]

    def add_edge(self, source: KnowledgeNode, target: KnowledgeNode,
                 relation: RelationType, metadata: Dict = None):
        edge = KnowledgeEdge(source, target, relation, metadata or {})
        self.edges.append(edge)
        logger.info(f"🔗 Edge: {source.id} --[{relation.value}]--> {target.id}")
        if self.tracker:
            self.tracker.on_edge(source.id, target.id, relation.value)

    def get_nodes_by_type(self, node_type: NodeType) -> List[KnowledgeNode]:
        return [n for n in self.nodes.values() if n.type == node_type]

    def log_search_step(self, step: str):
        self.search_path.append(step)
        logger.info(f"🔍 {step}")
        if self.tracker:
            self.tracker.log("search", step)

    def get_summary(self) -> str:
        datasets = self.get_nodes_by_type(NodeType.DATASET)
        papers   = self.get_nodes_by_type(NodeType.PAPER)
        models   = self.get_nodes_by_type(NodeType.MODEL)
        return f"""### Knowledge Graph Summary
**Search Path:**
{chr(10).join(f"{i+1}. {step}" for i, step in enumerate(self.search_path))}
**Resources Found:**
- 📊 Datasets: {len(datasets)}
- 📄 Papers: {len(papers)}
- 🤖 Models: {len(models)}
- 🔗 Connections: {len(self.edges)}
"""


# ============================================================
# DIRECT API CALLS
# ============================================================

def direct_dataset_search(query: str = None, tags: List[str] = None, limit: int = 25, hf_token=None) -> List[Dict]:
    api = HfApi(token=hf_token)
    results = []
    if tags:
        datasets = api.list_datasets(filter=tags, full=True, limit=limit)
    else:
        datasets = api.list_datasets(search=query, full=True, sort="downloads", limit=limit)
    for ds in datasets:
        dataset_id = ds.id
        meta_req = requests.get(f"https://huggingface.co/api/datasets/{dataset_id}")
        if meta_req.status_code != 200:
            continue
        meta = meta_req.json()
        all_tags = meta.get("tags", []) or []
        if tags:
            matches = [tag for tag in tags if tag in all_tags]
            if not matches:
                continue
        results.append({
            "id": dataset_id,
            "tags": all_tags,
            "downloads": getattr(ds, "downloads", 0),
            "likes": getattr(ds, "likes", 0),
            "description": meta.get("description", ""),
            "raw": meta,
        })
    logger.info(f"Found {len(results)} datasets")
    return results


def direct_model_search(query: str = None, tags: List[str] = None, limit: int = 25, hf_token=None) -> List[Dict]:
    api = HfApi(token=hf_token)
    results = []
    if tags:
        models = api.list_models(filter=tags, full=True, limit=limit)
    else:
        models = api.list_models(search=query, full=True, sort="downloads", limit=limit)
    for m in models:
        model_id = m.id
        meta_req = requests.get(f"https://huggingface.co/api/models/{model_id}")
        if meta_req.status_code != 200:
            continue
        meta = meta_req.json()
        all_tags = meta.get("tags", []) or []
        if tags and not any(tag in all_tags for tag in tags):
            continue
        results.append({
            "id": model_id,
            "tags": all_tags,
            "downloads": getattr(m, "downloads", 0),
            "likes": getattr(m, "likes", 0),
            "raw": meta,
        })
    return results


# ============================================================
# KNOWLEDGE GRAPH BUILDERS
# ============================================================

@dataclass
class TagCandidates:
    free_tags: List[str]
    task_tags: List[str]


def extract_candidate_tags(tags: Set[str]) -> TagCandidates:
    IGNORED = {"license:", "library:", "region:", "format:", "size_categories:", "modality:", "language:"}
    TASK_PREFIX = "task_categories:"
    free_tags, task_tags = [], []
    for tag in tags:
        if any(tag.startswith(p) for p in IGNORED):
            continue
        if tag.startswith(TASK_PREFIX):
            task_tags.append(tag.replace(TASK_PREFIX, ""))
        elif ":" not in tag:
            free_tags.append(tag)
    return TagCandidates(free_tags=sorted(set(free_tags)), task_tags=sorted(set(task_tags)))


def build_kg_from_datasets(initial_query: str, initial_datasets: List[Dict],
                            kg: KnowledgeGraph, min_datasets: int = 3, hf_token=None, llm=None, mcp_client=None) -> KnowledgeGraph:
    kg.log_search_step(f"Building KG from {len(initial_datasets)} initial datasets")
    dataset_nodes = []
    for ds_data in initial_datasets:
        node = KnowledgeNode(
            id=ds_data["id"], type=NodeType.DATASET,
            tags=set(ds_data.get("tags", [])),
            metadata={"downloads": ds_data.get("downloads", 0), "description": ds_data.get("description", "")}
        )
        kg.add_node(node)
        dataset_nodes.append(node)

    papers_found = False
    for dataset_node in dataset_nodes[:5]:
        if len(kg.get_nodes_by_type(NodeType.DATASET)) >= min_datasets:
            break
        arxiv_tags = [t for t in dataset_node.tags if t.startswith("arxiv:")]
        if arxiv_tags:
            kg.log_search_step(f"Found ArXiv tags in {dataset_node.id}: {arxiv_tags}")
            papers_found = True
            for arxiv_tag in arxiv_tags[:2]:
                paper_id = arxiv_tag.replace("arxiv:", "")
                try:
                    paper_summary = mcp_client.get_prompt("Paper Summary", {"paper_id": paper_id})
                    paper_node = KnowledgeNode(
                        id=paper_id, type=NodeType.PAPER,
                        metadata={"source": "arxiv_tag", "title": f"Paper {paper_id}",
                                  "summary": paper_summary or "", "has_content": bool(paper_summary)})
                except Exception as e:
                    paper_node = KnowledgeNode(
                        id=paper_id, type=NodeType.PAPER,
                        metadata={"source": "arxiv_tag", "title": f"Paper {paper_id}", "has_content": False})
                kg.add_node(paper_node)
                kg.add_edge(dataset_node, paper_node, RelationType.HAS_PAPER)
                if len(kg.get_nodes_by_type(NodeType.DATASET)) >= min_datasets:
                    break
                kg.log_search_step(f"Searching resources for paper {paper_id}")
                for ds_data in direct_dataset_search(tags=[arxiv_tag], limit=10, hf_token=hf_token):
                    if ds_data["id"] != dataset_node.id:
                        rel_node = KnowledgeNode(
                            id=ds_data["id"], type=NodeType.DATASET,
                            tags=set(ds_data.get("tags", [])),
                            metadata={"downloads": ds_data.get("downloads", 0), "description": ds_data.get("description", "")})
                        kg.add_node(rel_node)
                        kg.add_edge(paper_node, rel_node, RelationType.HAS_DATASET)
                for model_data in direct_model_search(tags=[arxiv_tag], limit=3, hf_token=hf_token):
                    model_node = KnowledgeNode(
                        id=model_data["id"], type=NodeType.MODEL,
                        tags=set(model_data.get("tags", [])),
                        metadata={"downloads": model_data.get("downloads", 0)})
                    kg.add_node(model_node)
                    kg.add_edge(paper_node, model_node, RelationType.HAS_MODEL)

    if not papers_found and len(kg.get_nodes_by_type(NodeType.DATASET)) < min_datasets:
        kg.log_search_step("Using tag-based search for expansion")
        for dataset_node in dataset_nodes[:9]:
            if len(kg.get_nodes_by_type(NodeType.DATASET)) >= min_datasets:
                break
            candidates = extract_candidate_tags(dataset_node.tags)
            relevant_tags = tag_selection_agent(initial_query, dataset_node.id, candidates.free_tags, llm, max_tags=2, tracker=kg.tracker)
            if not relevant_tags and candidates.task_tags:
                relevant_tags = tag_selection_agent(initial_query, dataset_node.id, candidates.task_tags, llm, max_tags=1, tracker=kg.tracker)
            if not relevant_tags:
                continue
            for tag_obj in relevant_tags:
                if len(kg.get_nodes_by_type(NodeType.DATASET)) >= min_datasets:
                    break
                tag = tag_obj["name"]
                kg.log_search_step(f"Searching via tag: {tag}")
                for ds_data in direct_dataset_search(tags=[tag], limit=10, hf_token=hf_token):
                    if ds_data["id"] != dataset_node.id:
                        rel_node = KnowledgeNode(
                            id=ds_data["id"], type=NodeType.DATASET,
                            tags=set(ds_data.get("tags", [])),
                            metadata={"downloads": ds_data.get("downloads", 0), "description": ds_data.get("description", "")})
                        kg.add_node(rel_node)
                        kg.add_edge(dataset_node, rel_node, RelationType.SHARES_TAG, {"tag": tag})
    return kg


def build_kg_from_papers(initial_query: str, paper_ids: list,
                          kg: KnowledgeGraph = None, paper_data: dict = None, hf_token=None, llm=None) -> KnowledgeGraph:
    if kg is None:
        kg = KnowledgeGraph()
    if paper_data is None:
        paper_data = {}
    kg.log_search_step(f"Building KG from {len(paper_ids)} papers")
    for pid in paper_ids:
        pdata = paper_data.get(pid, {})
        metadata = {
            "source": "paper_search",
            "title": pdata.get("title", ""),
            "abstract": pdata.get("abstract", ""),
            "ai_summary": pdata.get("ai_summary", ""),
            "ai_keywords": pdata.get("ai_keywords", ""),
            "authors": pdata.get("authors", ""),
            "published": pdata.get("published", ""),
            "link": pdata.get("link", ""),
            "full_text": pdata.get("full_text", ""),
            "has_content": bool(pdata.get("abstract") or pdata.get("ai_summary"))
        }
        paper_node = KnowledgeNode(id=pid, type=NodeType.PAPER, metadata=metadata)
        kg.add_node(paper_node)
        for ds in direct_dataset_search(query=pid, limit=5, hf_token=hf_token):
            ds_node = KnowledgeNode(id=ds["id"], type=NodeType.DATASET,
                                    tags=set(ds.get("tags", [])),
                                    metadata={"downloads": ds.get("downloads", 0), "description": ds.get("description", "")})
            kg.add_node(ds_node)
            kg.add_edge(paper_node, ds_node, RelationType.HAS_DATASET)
        for m in direct_model_search(query=pid, limit=3, hf_token=hf_token):
            model_node = KnowledgeNode(id=m["id"], type=NodeType.MODEL,
                                       tags=set(m.get("tags", [])),
                                       metadata={"downloads": m.get("downloads", 0)})
            kg.add_node(model_node)
            kg.add_edge(paper_node, model_node, RelationType.HAS_MODEL)
    return kg


def fallback_search(user_message, history, initial_datasets, llm, tools, mcp_client, hf_token=None,
                     tracker: LogTreeTracker = None):
    if tracker:
        tracker.log("fallback", "Running fallback search strategy…")
    try:
        paper_ids, paper_data = semantic_paper_search_agent(user_message, history, llm, tools, mcp_client, tracker)
        if paper_ids:
            kg = KnowledgeGraph(tracker)
            return build_kg_from_papers(user_message, paper_ids, kg, paper_data, hf_token=hf_token)
    except Exception as e:
        logger.error(f"Paper fallback failed: {e}")
    try:
        tags = tag_matching_agent(user_message, history, llm, tracker)
        if tags:
            datasets = []
            for tag in tags:
                datasets.extend(direct_dataset_search(query=tag, limit=5, hf_token=hf_token))
            if datasets:
                kg = KnowledgeGraph(tracker)
                return build_kg_from_datasets(user_message, datasets, kg, hf_token=hf_token)
    except Exception as e:
        logger.error(f"Tag fallback failed: {e}")
    kg = KnowledgeGraph(tracker)
    return build_kg_from_datasets(user_message, initial_datasets[:5], kg, hf_token=hf_token)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def compute_hybrid_score(ds: KnowledgeNode):
    """Compute hybrid score for dataset ranking"""
    downloads = float(ds.metadata.get("downloads", 0))
    likes = float(ds.metadata.get("likes", 0))
    confidence = float(ds.metadata.get("confidence", 0.5))
    score = (downloads * 0.4 + likes * 0.3 + confidence * 100 * 0.3)
    ds.metadata["rank_score"] = score


def pre_rank_datasets(datasets: List[KnowledgeNode], top_k: int = 5) -> List[KnowledgeNode]:
    """Pre-rank datasets by hybrid score"""
    return sorted(datasets, key=lambda d: d.metadata.get("rank_score", 0), reverse=True)[:top_k]


def enrich_with_details(datasets: List[KnowledgeNode], mcp_client: MCPPromptClient) -> Dict[str, str]:
    """Enrich datasets with detailed information"""
    enriched = {}
    for ds in datasets:
        try:
            details = mcp_client.get_prompt("Dataset Details", {"dataset_id": ds.id})
            if details:
                enriched[ds.id] = details
        except Exception:
            enriched[ds.id] = f"Dataset: {ds.id}"
    return enriched


# ============================================================
# MAIN STREAMING FUNCTION
# ============================================================

def ask_llm_streaming(user_message: str, history: list, hf_token:str, theme: str = "dark"):
    llm = InferenceClient(
        model="openai/gpt-oss-20b",
        token=hf_token,
    )

    mcp_client = MCPPromptClient(MCP_SERVER, hf_token)
    """
    Generator that yields (tree_html, chat_response) tuples.
    The tree updates live while the chat response fills in at the end.
    """
    tracker = LogTreeTracker()
    tracker.log("info", f"Query received: {user_message[:80]}…")

    yield tracker.render_tree_svg(theme), ""

    try:
        messages = list(history) + [{"role": "user", "content": user_message}]
        kg = KnowledgeGraph(tracker)
        kg_triggered = False
        initial_response = ""

        # ── 1. Tool selection ────────────────────────────────
        tracker.log("search", "Selecting tools for query…")
        yield tracker.render_tree_svg(theme), ""

        stream = query_understanding_agent(user_message, history, llm, tools)
        tool_calls = []
        blocked_call = False

        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                initial_response += delta.content
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    raw_name = tc.function.name
                    name = re.sub(r"<\|.*?\|>.*$", "", raw_name)
                    try:
                        args = json.loads(tc.function.arguments)
                    except Exception:
                        args = {}
                    if "hub_repo_details" in name and args.get("repo_type") == "paper":
                        blocked_call = True
                        break
                    tool_calls.append({"name": name, "args": args})
            if blocked_call:
                break

        # ── 2. Dataset search ────────────────────────────────
        dataset_calls = [tc for tc in tool_calls if "hub_repo_search" in tc["name"]]
        all_found_datasets = {}

        for idx, ds_call in enumerate(dataset_calls):
            args = ds_call.get("args", {})
            limit = args.get("limit", 10)
            queries = args.get("queries") or ([args.get("query")] if args.get("query") else [])
            for q in queries:
                if not q:
                    continue
                tracker.log("search", f"Searching datasets: {q}")
                yield tracker.render_tree_svg(theme), ""
                try:
                    results = direct_dataset_search(query=q, limit=limit, hf_token=hf_token)
                    for ds in results:
                        all_found_datasets[ds["id"]] = ds
                    tracker.log("dataset", f"Found {len(results)} results for {q}")
                    yield tracker.render_tree_svg(theme), ""
                except Exception as e:
                    tracker.log("error", f"Dataset search failed: {e}")
                    yield tracker.render_tree_svg(theme), ""

        if all_found_datasets:
            kg = build_kg_from_datasets(user_message, list(all_found_datasets.values()), kg, hf_token=hf_token)
            kg_triggered = True
            yield tracker.render_tree_svg(theme), ""

        # ── 3. Paper fallback ────────────────────────────────
        if len(kg.get_nodes_by_type(NodeType.DATASET)) < 3:
            tracker.log("fallback", "Few datasets found – trying paper search…")
            yield tracker.render_tree_svg(theme), ""
            paper_ids, paper_data = semantic_paper_search_agent(
                user_message, history, llm, tools, mcp_client, tracker)
            if paper_ids:
                kg = build_kg_from_papers(user_message, paper_ids, kg, paper_data, hf_token=hf_token)
                kg_triggered = True
                yield tracker.render_tree_svg(theme), ""

        # ── 4. Tag fallback ──────────────────────────────────
        if len(kg.get_nodes_by_type(NodeType.DATASET)) < 3:
            tracker.log("fallback", "Still few datasets – trying tag matching…")
            yield tracker.render_tree_svg(theme), ""
            candidate_tags = tag_matching_agent(user_message, history, llm, tracker)
            tag_datasets = {}
            for tag in candidate_tags:
                try:
                    results = direct_dataset_search(tags=[tag], limit=5, hf_token=hf_token)
                    for ds in results:
                        tag_datasets[ds["id"]] = ds
                except Exception:
                    pass
            if tag_datasets:
                kg = build_kg_from_datasets(user_message, list(tag_datasets.values()), kg, hf_token=hf_token)
                kg_triggered = True
                yield tracker.render_tree_svg(theme), ""

        # ── 5. Synthesis ─────────────────────────────────────
        if kg_triggered and kg and kg.nodes:
            tracker.log("filter", "Judging relevance of candidates…")
            yield tracker.render_tree_svg(theme), ""

            candidates = kg.get_nodes_by_type(NodeType.DATASET)
            cheap_filtered = prefilter_judgement_agent(user_message, candidates, llm, tracker)
            yield tracker.render_tree_svg(theme), ""

            CONF_THRESHOLD = 0.60
            sure_nodes, unsure_nodes = [], []
            for ds in cheap_filtered:
                conf = float(ds.metadata.get("confidence", 0.0))
                if ds.metadata.get("relevant") == "yes" and conf >= CONF_THRESHOLD:
                    sure_nodes.append(ds)
                elif ds.metadata.get("relevant") == "yes":
                    unsure_nodes.append(ds)

            tracker.log("filter", f"High-confidence: {len(sure_nodes)} | Borderline: {len(unsure_nodes)}")
            yield tracker.render_tree_svg(theme), ""

            judged_unsure = []
            if unsure_nodes:
                judged_unsure = full_context_relevance_judgement_agent(
                    user_message, unsure_nodes, mcp_client, llm, tracker)
                yield tracker.render_tree_svg(theme), ""

            judged_datasets = sure_nodes + judged_unsure

            # Paper fallback if no datasets survived
            if not judged_datasets:
                paper_nodes = kg.get_nodes_by_type(NodeType.PAPER)
                if paper_nodes:
                    tracker.log("warn", "No datasets survived – checking papers…")
                    yield tracker.render_tree_svg(theme), ""
                    papers_for_j = [{"paper_id": p.id, "metadata": p.metadata} for p in paper_nodes]
                    judged_papers = paper_relevance_judgement_agent(user_message, papers_for_j, llm, tracker)
                    yield tracker.render_tree_svg(theme), ""

                    if judged_papers:
                        top_papers = sorted(judged_papers, key=lambda x: float(x.get("confidence", 0)), reverse=True)[:5]
                        enriched_papers = {}
                        for p in top_papers:
                            try:
                                summary = mcp_client.get_prompt("Paper Summary", {"paper_id": p["paper_id"]})
                                if summary:
                                    enriched_papers[p["paper_id"]] = summary
                            except Exception:
                                pass
                        blocks = []
                        for p in top_papers:
                            details = enriched_papers.get(p["paper_id"], "No details")[:800]
                            blocks.append(f"=== PAPER ===\nID: {p['paper_id']}\n{details}\n=== END ===")
                        synthesis_prompt = f"""No datasets found, but relevant papers exist.
Query: {user_message}
Papers:
{chr(10).join(blocks)}
Provide: statement about missing datasets, paper summaries, HF links, suggestions."""
                        messages.append({"role": "assistant", "content": initial_response})
                        messages.append({"role": "user", "content": synthesis_prompt})
                        synthesis = llm.chat.completions.create(messages=messages, max_tokens=1800, temperature=0.6)
                        tracker.log("done", "Answer ready (papers only)")
                        yield tracker.render_tree_svg(theme), synthesis.choices[0].message.content
                        return

                tracker.log("error", "No relevant datasets or papers found")
                yield tracker.render_tree_svg(theme), "I couldn't find relevant datasets or papers. Please try rephrasing."
                return

            # Re-rank and synthesise
            tracker.log("rank", f"Re-ranking {len(judged_datasets)} final candidates…")
            yield tracker.render_tree_svg(theme), ""

            for ds in judged_datasets:
                compute_hybrid_score(ds)
            top_candidates = pre_rank_datasets(judged_datasets, top_k=5)
            enriched = enrich_with_details(top_candidates, mcp_client)

            try:
                final_ranking = reranking_agent(user_message, top_candidates, enriched, llm, tracker)
            except Exception as e:
                logger.error(f"Reranking failed: {e}")
                final_ranking = [{"name": ds.id, "rank": i+1, "reason": "pre-rank fallback"}
                                  for i, ds in enumerate(top_candidates)]

            yield tracker.render_tree_svg(theme), ""

            ranked_datasets = []
            for item in sorted(final_ranking, key=lambda x: x["rank"]):
                ds = next((d for d in top_candidates if d.id == item["name"]), None)
                if ds:
                    ranked_datasets.append(ds)

            papers = kg.get_nodes_by_type(NodeType.PAPER)[:2]
            for paper in papers:
                try:
                    summary = mcp_client.get_prompt("Paper Summary", {"paper_id": paper.id})
                    if summary:
                        enriched[paper.id] = summary
                except Exception:
                    pass

            blocks = []
            for node in ranked_datasets:
                details = enriched.get(node.id, "No details")[:800]
                blocks.append(f"=== DATASET ===\nID: {node.id}\n{details}\n=== END ===")
            for paper in papers:
                details = enriched.get(paper.id, "No details")[:800]
                blocks.append(f"=== PAPER ===\nID: {paper.id}\n{details}\n=== END ===")

            synthesis_prompt = f"""Answer based on this Knowledge Graph.
Query: {user_message}
{chr(10).join(blocks)}
Provide: direct answer, dataset/paper recommendations with HF links, code examples."""

            messages.append({"role": "assistant", "content": initial_response})
            messages.append({"role": "user", "content": synthesis_prompt})

            tracker.log("done", "Generating final answer…")
            yield tracker.render_tree_svg(theme), ""

            synthesis = llm.chat.completions.create(messages=messages, max_tokens=1800, temperature=0.6)
            final_answer = synthesis.choices[0].message.content

            tracker.log("done", "✅ Complete!")
            yield tracker.render_tree_svg(theme), final_answer
            return

        tracker.log("warn", "KG empty – no results found")
        yield tracker.render_tree_svg(theme), "I couldn't find relevant datasets or papers."

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        tracker.log("error", f"Fatal error: {e}")
        yield tracker.render_tree_svg(theme), f"❌ Error: {str(e)}"


# ============================================================
# WALKTHROUGH STEPS
# ============================================================

WALKTHROUGH_STEPS = [
    {
        "title": "Welcome to DatasetChatty 🐸",
        "content": "Your AI-powered research assistant for discovering datasets, papers, and models on Hugging Face Hub. Let me show you around!",
        "selector": None,
        "position": "center",
    },
    {
        "title": "Step 1 — Secure Your Token 🔑",
        "content": "Paste your Hugging Face API token (starts with hf_) and click Save. This enables access to Hub resources.",
        "selector": "#token-status",
        "position": "left",
        "openSidebar": True,
        "interactive": True,
        "expandSpotlight": {"top": 170, "bottom": 0, "left": 10, "right": 10},
    },
    {
        "title": "Ask Your Research Question 💭",
        "content": "Type any research query: 'sentiment analysis datasets', 'audio classification models', 'recent NLP papers'...",
        "selector": "#input-container",
        "position": "top",
    },
    {
        "title": "Watch the Knowledge Graph 🧠",
        "content": "See my thinking process in real-time. Every search step, ranking decision, and discovery appears here.",
        "selector": "#tree-panel-container",
        "position": "right",
    },
    {
        "title": "Get Curated Recommendations 📚",
        "content": "High-quality results with direct links and code snippets appear here instantly.",
        "selector": "#chat-panel",
        "position": "bottom",
    },
    {
        "title": "You're Ready! ✨",
        "content": "Click Froggy 🐸 anytime to restart this tour. Now explore the vast world of datasets and models!",
        "selector": None,
        "position": "center",
    },
]

# ============================================================
# UNIFIED CSS WITH DARK/LIGHT MODE 
# ============================================================

CSS_DARK = """
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Poppins:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

/* ── Toast Notification ── */
#token-toast {
    position: fixed;
    bottom: 20px;
    right: 20px;
    background: #ff5252;
    color: white;
    padding: 12px 18px;
    border-radius: 8px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    opacity: 0;
    transform: translateY(20px);
    transition: opacity 0.3s, transform 0.3s;
    z-index: 9999;
}

#token-toast.show {
    opacity: 1;
    transform: translateY(0);
}
#token-toast-close {
    background: none; border: none; color: #fecaca; font-size: 20px;
    cursor: pointer; padding: 0; margin-left: auto; transition: all 0.2s;
}
#token-toast-close:hover { transform: scale(1.2); }
@keyframes slideUp {
    from { opacity: 0; transform: translateY(20px) translateX(20px) scale(0.9); }
    to { opacity: 1; transform: translateY(0) translateX(0) scale(1); }
}
body.light-mode #token-toast {
    background: linear-gradient(135deg, rgba(220, 38, 38, 0.9), rgba(185, 28, 28, 0.9));
    border-color: rgba(185, 28, 28, 0.5); color: #fee2e2;
}

:root {
    /* DARK MODE - UNIFIED */
    --bg-dark: #0f1419;
    --bg-secondary: #161b23;
    --bg-tertiary: #1a2028;
    --bg-glass: rgba(22, 27, 35, 0.7);
    --border-glass: rgba(0, 217, 255, 0.15);
    --text-primary: #e0f2fe;
    --text-secondary: #cbd5e1;
    --text-muted: #94a3b8;
    --accent-cyan: #00d9ff;
    --accent-purple: #c084fc;
    --accent-pink: #f472b6;
    --accent-glow-cyan: 0 0 20px rgba(0, 217, 255, 0.5);
    --accent-glow-purple: 0 0 20px rgba(192, 132, 252, 0.4);
    --shadow-sm: 0 2px 8px rgba(0, 0, 0, 0.3);
    --shadow-md: 0 8px 24px rgba(0, 0, 0, 0.4);
    --shadow-lg: 0 16px 40px rgba(0, 0, 0, 0.5);
    
    /* Chat panels - slightly darker in dark mode for unity */
    --chat-bg: #161b23;
    --chat-bg-secondary: #1a2028;
    --chat-text-primary: white;
    --chat-text-secondary: white;
    --chat-border: rgba(0, 217, 255, 0.15);
    --chat-user-bg: rgba(0, 217, 255, 0.1);
    --chat-bot-bg: rgba(192, 132, 252, 0.08);
    --chat-user-border: rgba(0, 217, 255, 0.3);
    --chat-bot-border: rgba(192, 132, 252, 0.2);
}

body.light-mode {
    /* LIGHT MODE OVERRIDES */
    --bg-dark: #f0f7ff;
    --bg-secondary: #e3f2fd;
    --bg-tertiary: #bbdefb;
    --bg-glass: rgba(227, 242, 253, 0.8);
    --border-glass: rgba(30, 144, 255, 0.3);
    --text-primary: #01579b;
    --text-secondary: #0277bd;
    --text-muted: #01579b;
    --accent-cyan: #0277bd;
    --accent-purple: #1565c0;
    --accent-pink: #0288d1;
    --accent-glow-cyan: 0 0 20px rgba(2, 119, 189, 0.3);
    --accent-glow-purple: 0 0 20px rgba(21, 101, 192, 0.3);
    --shadow-sm: 0 2px 8px rgba(0, 0, 0, 0.08);
    --shadow-md: 0 8px 24px rgba(0, 0, 0, 0.1);
    --shadow-lg: 0 16px 40px rgba(0, 0, 0, 0.12);
    
    /* Chat panels remain light */
    --chat-bg: #ffffff;
    --chat-bg-secondary: #f0f7ff;
    --chat-text-primary: #01579b;
    --chat-text-secondary: #0277bd;
    --chat-border: #64b5f6;
    --chat-user-bg: #bbdefb;
    --chat-bot-bg: #e3f2fd;
    --chat-user-border: #0277bd;
    --chat-bot-border: #64b5f6;
}



.gradio-container, .gradio-container > .gradio-block {
    width: 100% !important;
    max-width: 100% !important;
    margin: 0 !important;
    padding: 0 !important;
}

#main-container {
    width: 100% !important;
    max-width: 100% !important;
    padding: 0 24px; /* optional für Innenabstand */
    margin: 0 auto;
}

html, body {
    height: 100%;
    margin: 0;
    padding: 0;
}
* { font-family: 'Poppins', sans-serif !important; }
body, .gradio-container { 
    background: linear-gradient(135deg, var(--bg-dark) 0%, var(--bg-tertiary) 50%, var(--bg-dark) 100%) !important;
    color: var(--text-primary) !important;
    min-height: 100vh;
    transition: background 0.4s ease, color 0.4s ease;
}
.gradio-container { max-width: 100% !important; margin: 0 auto !important; padding: 20px !important; }
footer { display: none !important; }

/* ── Animated background particles ── */
body::before {
    content: '';
    position: fixed;
    inset: 0;
    background: radial-gradient(circle at 20% 50%, rgba(0, 217, 255, 0.02) 0%, transparent 50%),
                radial-gradient(circle at 80% 80%, rgba(192, 132, 252, 0.02) 0%, transparent 50%);
    pointer-events: none;
    z-index: 1;
    animation: float 20s ease-in-out infinite;
    transition: background 0.4s ease;
}

body.light-mode::before {
    background: radial-gradient(circle at 20% 50%, rgba(2, 119, 189, 0.08) 0%, transparent 50%),
                radial-gradient(circle at 80% 80%, rgba(21, 101, 192, 0.08) 0%, transparent 50%);
}

@keyframes float {
    0%, 100% { transform: translateY(0) translateX(0); }
    50% { transform: translateY(-20px) translateX(10px); }
}

/* ── Header ── */
#header-container {
    background: var(--bg-glass);
    backdrop-filter: blur(12px);
    border: 1px solid var(--border-glass);
    border-radius: 20px;
    padding: 24px 32px;
    margin-bottom: 24px;
    box-shadow: var(--shadow-md);
    display: flex;
    align-items: center;
    gap: 20px;
    animation: slideDown 0.8s cubic-bezier(0.34, 1.56, 0.64, 1);
    position: relative;
    z-index: 2;
    transition: all 0.4s ease;
}

@keyframes slideDown {
    from { opacity: 0; transform: translateY(-30px); }
    to { opacity: 1; transform: translateY(0); }
}

#header-container::before {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(90deg, transparent, rgba(0, 217, 255, 0.08), transparent);
    border-radius: 20px;
    animation: shine 3s ease-in-out infinite;
    pointer-events: none;
}

body.light-mode #header-container::before {
    background: linear-gradient(90deg, transparent, rgba(2, 119, 189, 0.12), transparent);
}

@keyframes shine {
    0%, 100% { opacity: 0; }
    50% { opacity: 1; }
}

#froggy-btn {
    font-size: 40px !important;
    background: linear-gradient(135deg, rgba(0, 217, 255, 0.15), rgba(192, 132, 252, 0.15)) !important;
    border: 2px solid var(--accent-cyan) !important;
    cursor: pointer !important;
    padding: 12px !important;
    border-radius: 16px !important;
    transition: all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1) !important;
    min-width: auto !important;
    box-shadow: var(--accent-glow-cyan) !important;
}
#froggy-btn:hover {
    background: linear-gradient(135deg, rgba(0, 217, 255, 0.3), rgba(192, 132, 252, 0.3)) !important;
    transform: scale(1.15) rotate(-8deg);
    box-shadow: 0 0 30px rgba(0, 217, 255, 0.6) !important;
}

body.light-mode #froggy-btn {
    background: linear-gradient(135deg, rgba(2, 119, 189, 0.1), rgba(21, 101, 192, 0.1)) !important;
    border-color: var(--accent-cyan) !important;
}

body.light-mode #froggy-btn:hover {
    background: linear-gradient(135deg, rgba(2, 119, 189, 0.25), rgba(21, 101, 192, 0.25)) !important;
}

#header-container h2, #header-container p {
    margin: 0 !important;
    color: var(--text-primary) !important;
}

/* ── Theme Toggle Button ── */
#theme-toggle-btn {
    background: linear-gradient(135deg, var(--accent-cyan), var(--accent-purple)) !important;
    border: none !important;
    color: white !important;
    border-radius: 12px !important;
    padding: 10px 16px !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    cursor: pointer !important;
    box-shadow: var(--accent-glow-cyan) !important;
    transition: all 0.3s ease !important;
    margin-left: auto !important;
}

#theme-toggle-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 0 25px rgba(0, 217, 255, 0.5) !important;
}

body.light-mode #theme-toggle-btn {
    background: linear-gradient(135deg, #0277bd, #1565c0) !important;
    box-shadow: 0 0 20px rgba(2, 119, 189, 0.25) !important;
}

/* ── Floating Froggy guide ── */
#froggy-guide {
    position: fixed;
    font-size: 44px;
    z-index: 10001;
    pointer-events: none;
    transition: top 0.6s cubic-bezier(0.34, 1.56, 0.64, 1),
                left 0.6s cubic-bezier(0.34, 1.56, 0.64, 1),
                opacity 0.3s ease;
    opacity: 0;
    filter: drop-shadow(0 8px 16px rgba(0, 217, 255, 0.3));
}
#froggy-guide.visible { opacity: 1; }
#froggy-guide.hopping { animation: frogHop 0.6s cubic-bezier(0.34, 1.56, 0.64, 1); }
@keyframes frogHop {
    0%   { transform: translateY(0) scale(1) rotate(0deg); }
    40%  { transform: translateY(-32px) scale(1.2) rotate(-12deg); }
    70%  { transform: translateY(-8px) scale(0.95) rotate(6deg); }
    100% { transform: translateY(0) scale(1) rotate(0deg); }
}

/* ── Walkthrough overlay ── */
#walkthrough-overlay {
    position: fixed;
    inset: 0;
    z-index: 9998;
    pointer-events: none;
    opacity: 0;
    transition: opacity 0.3s ease;
    background: rgba(15, 20, 25, 0.85);
    backdrop-filter: blur(4px);
}
#walkthrough-overlay.active {
    opacity: 1;
    pointer-events: auto;
}

body.light-mode #walkthrough-overlay {
    background: rgba(240, 247, 255, 0.75);
}

/* ── Tooltip ── */
#walkthrough-tooltip {
    position: fixed;
    background: var(--bg-glass);
    backdrop-filter: blur(12px);
    border-radius: 20px;
    padding: 32px;
    box-shadow: var(--shadow-lg);
    border: 2px solid var(--border-glass);
    z-index: 10000;
    width: 340px;
    opacity: 0;
    transform: translateY(12px) scale(0.95);
    pointer-events: none;
    transition: opacity 0.25s ease, transform 0.25s ease;
}
#walkthrough-tooltip.active {
    opacity: 1;
    transform: translateY(0) scale(1);
    pointer-events: auto;
}
#walkthrough-tooltip h3 {
    margin: 0 0 14px;
    color: var(--accent-cyan);
    font-size: 19px;
    font-weight: 700;
    text-shadow: 0 0 10px rgba(0, 217, 255, 0.5);
}
#walkthrough-tooltip p {
    margin: 0 0 20px;
    color: var(--text-secondary);
    font-size: 14px;
    line-height: 1.7;
}
.step-indicator { display: flex; gap: 8px; margin-bottom: 20px; }
.step-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--text-muted);
    transition: all 0.25s;
    box-shadow: 0 0 8px rgba(0, 217, 255, 0.2);
}
.step-dot.active { 
    background: var(--accent-cyan);
    transform: scale(1.4);
    box-shadow: var(--accent-glow-cyan);
}
.wt-buttons { display: flex; gap: 10px; justify-content: flex-end; }
.wt-buttons button {
    padding: 10px 20px;
    border-radius: 12px;
    font-size: 13px; font-weight: 600;
    cursor: pointer; transition: all 0.2s;
    border: none;
    font-family: 'Poppins', sans-serif;
}
.skip-btn {
    background: transparent;
    border: 1.5px solid var(--accent-cyan) !important;
    color: var(--accent-cyan);
}
.skip-btn:hover { 
    background: rgba(0, 217, 255, 0.15);
    box-shadow: var(--accent-glow-cyan);
}
.next-btn { 
    background: linear-gradient(135deg, var(--accent-cyan), var(--accent-purple));
    color: white;
    box-shadow: var(--accent-glow-cyan);
}
.next-btn:hover { 
    transform: translateY(-2px);
    box-shadow: 0 0 30px rgba(0, 217, 255, 0.6);
}

/* ── Main layout ── */
#main-container { 
    display: flex; 
    flex-direction: column;
    gap: 24px;
    animation: fadeIn 0.8s ease-out 0.2s backwards;
}

@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}

/* ── CHAT PANEL - UNIFIED ── */
#chat-panel {
    flex: 1.2;
    background: var(--chat-bg);
    border-radius: 20px;
    border: 1px solid var(--chat-border);
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
    overflow: hidden;
    display: flex;
    flex-direction: column;
    transition: all 0.3s ease;
}

#chat-panel:hover {
    border-color: var(--chat-border);
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.25);
}

#chatbot {
    flex: 1;
    min-height: 480px;
    border: none !important;
    background: transparent !important;
}

#chatbot .message {
    padding: 14px 18px !important;
    border-radius: 14px !important;
    font-size: 14px !important;
    line-height: 1.6 !important;
    margin: 8px 0 !important;
    animation: messageSlide 0.4s ease-out;
}

@keyframes messageSlide {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

#chatbot .message.user { 
    background: var(--chat-user-bg) !important;
    border: 2px solid var(--chat-user-border) !important;
    margin-left: 40px !important;
    color: var(--chat-text-primary) !important;
}

#chatbot .message.bot { 
    background: var(--chat-bot-bg) !important;
    border: 2px solid var(--chat-bot-border) !important;
    margin-right: 40px !important;
    color: var(--chat-text-primary) !important;
}

#chatbot .message.user * {
    color: var(--chat-text-primary) !important;
}

#chatbot .message.bot * {
    color: var(--chat-text-primary) !important;
}

#input-container {
    padding: 20px;
    border-top: 1px solid var(--chat-border);
    background: var(--chat-bg);
    border-radius: 0 0 20px 20px;
    transition: all 0.3s ease;
}

#msg-input textarea {
    border: 2px solid var(--chat-border) !important;
    border-radius: 14px !important;
    padding: 16px 18px !important;
    font-size: 14px !important;
    background: var(--bg-secondary) !important;
    color: var(--chat-text-primary) !important;
    transition: all 0.3s !important;
    font-family: 'Poppins', sans-serif !important;
}

#msg-input textarea::placeholder {
    color: var(--text-muted) !important;
}

#msg-input textarea:focus {
    border-color: var(--accent-cyan) !important;
    box-shadow: 0 0 20px rgba(0, 217, 255, 0.2) !important;
    background: var(--bg-secondary) !important;
}

body.light-mode #msg-input textarea:focus {
    border-color: #0277bd !important;
    box-shadow: 0 0 20px rgba(2, 119, 189, 0.15) !important;
}

#send-btn {
    background: linear-gradient(135deg, var(--accent-cyan), var(--accent-purple)) !important;
    border: none !important;
    border-radius: 14px !important;
    color: white !important;
    font-weight: 700 !important;
    padding: 0 28px !important;
    min-height: 50px !important;
    box-shadow: var(--accent-glow-cyan) !important;
    transition: all 0.3s !important;
    font-family: 'Poppins', sans-serif !important;
    cursor: pointer !important;
}

#send-btn:hover { 
    transform: translateY(-3px);
    box-shadow: 0 0 25px rgba(0, 217, 255, 0.5) !important;
}

#send-btn:active {
    transform: translateY(-1px);
}

body.light-mode #send-btn {
    background: linear-gradient(135deg, #0277bd, #0288d1) !important;
    box-shadow: 0 4px 12px rgba(2, 119, 189, 0.2) !important;
}

#new-chat-btn {
    background: transparent !important;
    border: 2px solid var(--accent-cyan) !important;
    border-radius: 14px !important;
    color: var(--accent-cyan) !important;
    font-weight: 600 !important;
    padding: 0 20px !important;
    min-height: 50px !important;
    box-shadow: none !important;
    transition: all 0.3s !important;
    font-family: 'Poppins', sans-serif !important;
    cursor: pointer !important;
}

#new-chat-btn:hover {
    border-color: var(--accent-cyan) !important;
    background: rgba(0, 217, 255, 0.1) !important;
    box-shadow: 0 4px 12px rgba(0, 217, 255, 0.15) !important;
}

body.light-mode #new-chat-btn {
    border-color: #0277bd !important;
    color: #0277bd !important;
}

body.light-mode #new-chat-btn:hover {
    background: rgba(2, 119, 189, 0.08) !important;
    box-shadow: 0 4px 12px rgba(2, 119, 189, 0.12) !important;
}

/* ── TREE PANEL - UNIFIED ── */
#tree-panel-container {
    flex: 1;
    background: var(--bg-glass);
    backdrop-filter: blur(12px);
    border-radius: 20px;
    border: 1px solid var(--border-glass);
    box-shadow: var(--shadow-md);
    overflow: hidden;
    display: flex;
    flex-direction: column;
    transition: all 0.3s ease;
}

#tree-panel-container:hover {
    border-color: rgba(0, 217, 255, 0.25);
    box-shadow: 0 0 30px rgba(0, 217, 255, 0.1);
}

/* ── FULLSCREEN MODE ── */
#tree-panel-container.fullscreen {
    position: fixed !important;
    inset: 0 !important;
    width: 100% !important;
    height: 100% !important;
    z-index: 10005 !important;
    border-radius: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    background: var(--bg-dark) !important;
}

#tree-panel-container.fullscreen #tree-header {
    flex-shrink: 0;
    position: relative;
}

#tree-panel-container.fullscreen #tree-close-btn {
    display: inline-block !important;
    position: absolute;
    right: 20px;
    top: 50%;
    transform: translateY(-50%);
}

#tree-panel-container.fullscreen #tree-fullscreen-btn {
    display: none !important;
}

#tree-panel-container.fullscreen #tree-panel {
    flex: 1 !important;
    height: auto !important;
    min-height: 0 !important;
    overflow-y: auto !important;
    padding: 20px !important;
}

#tree-header {
    padding: 20px 24px;
    border-bottom: 1px solid var(--border-glass);
    background: var(--bg-secondary);
    backdrop-filter: blur(8px);
    transition: all 0.3s ease;
}

#tree-header h3 { 
    margin: 0; 
    font-size: 16px; 
    font-weight: 700; 
    color: var(--text-primary);
    background: linear-gradient(90deg, var(--accent-cyan), var(--accent-purple));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

#tree-panel { 
    height: 100%; 
    overflow-y: auto;
    padding: 12px;
}

#tree-panel::-webkit-scrollbar {
    width: 8px;
}

#tree-panel::-webkit-scrollbar-track {
    background: transparent;
}

#tree-panel::-webkit-scrollbar-thumb {
    background: rgba(0, 217, 255, 0.2);
    border-radius: 4px;
    transition: background 0.3s;
}

#tree-panel::-webkit-scrollbar-thumb:hover {
    background: rgba(0, 217, 255, 0.4);
}

body.light-mode #tree-panel::-webkit-scrollbar-thumb {
    background: rgba(2, 119, 189, 0.2);
}

body.light-mode #tree-panel::-webkit-scrollbar-thumb:hover {
    background: rgba(2, 119, 189, 0.4);
}

/* ── Sidebar ── */
.sidebar { 
    backdrop-filter: blur(12px) !important;
    transition: all 0.3s ease;
}

.sidebar section {
    background: var(--bg-glass) !important;
    border: 1px solid var(--border-glass) !important;
    border-radius: 16px !important;
    padding: 20px !important;
    margin-bottom: 16px !important;
    transition: all 0.3s ease;
}

.sidebar h4 { 
    color: var(--text-primary) !important;
    transition: color 0.3s ease;
}

#token-status { 
    padding: 12px 16px; 
    border-radius: 12px; 
    font-size: 13px; 
    margin-top: 12px;
    font-weight: 600;
}

#token-status.valid { 
    background: rgba(16, 185, 129, 0.15);
    border: 1px solid rgba(16, 185, 129, 0.3);
    color: #6ee7b7;
}

#token-status.invalid { 
    background: rgba(239, 68, 68, 0.15);
    border: 1px solid rgba(239, 68, 68, 0.3);
    color: #fca5a5;
}

#token-status.pending { 
    background: rgba(0, 217, 255, 0.08);
    border: 1px solid rgba(0, 217, 255, 0.25);
    color: var(--accent-cyan);
}

body.light-mode #token-status.pending {
    background: rgba(2, 119, 189, 0.08);
    border: 1px solid rgba(2, 119, 189, 0.25);
    color: #0277bd;
}

#stats-section { padding: 0 !important; }

.stat-card { 
    background: var(--bg-tertiary);
    border: 1px solid var(--border-glass);
    border-radius: 12px; 
    padding: 14px 16px; 
    margin-bottom: 10px;
    transition: all 0.3s ease;
}

.stat-card:hover {
    border-color: rgba(0, 217, 255, 0.2);
    background: var(--bg-secondary);
}

.stat-card label { 
    font-size: 11px; 
    text-transform: uppercase; 
    letter-spacing: 0.8px; 
    color: var(--text-muted);
    margin-bottom: 6px; 
    display: block;
    font-weight: 600;
}

.stat-card value { 
    font-size: 15px; 
    font-weight: 700; 
    color: var(--accent-cyan);
    font-family: 'JetBrains Mono', monospace;
}

/* ── Spotlight ring ── */
.wt-spotlight {
    outline: 3px solid var(--accent-cyan) !important;
    outline-offset: 6px !important;
    border-radius: 20px !important;
    position: relative !important;
    z-index: 9999 !important;
    box-shadow: 0 0 30px rgba(0, 217, 255, 0.3) !important;
}

/* ── Button styling ── */
button {
    font-family: 'Poppins', sans-serif !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
}

.primary {
    background: linear-gradient(135deg, var(--accent-cyan), var(--accent-purple)) !important;
    border: none !important;
    box-shadow: var(--accent-glow-cyan) !important;
    color: #ffffff !important;
}

.primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 0 30px rgba(0, 217, 255, 0.5) !important;
}

/* ── Gradio text overrides ── */
.textbox input,
.textbox textarea {
    background: var(--bg-secondary) !important;
    border: 2px solid var(--chat-border) !important;
    color: var(--chat-text-primary) !important;
    border-radius: 12px !important;
    transition: all 0.3s !important;
    font-family: 'Poppins', sans-serif !important;
}

.textbox input::placeholder,
.textbox textarea::placeholder {
    color: var(--text-muted) !important;
}

.textbox input:focus,
.textbox textarea:focus {
    border-color: var(--accent-cyan) !important;
    box-shadow: 0 0 20px rgba(0, 217, 255, 0.15) !important;
}

body.light-mode .textbox input:focus,
body.light-mode .textbox textarea:focus {
    border-color: #0277bd !important;
    box-shadow: 0 0 20px rgba(2, 119, 189, 0.15) !important;
}

.textbox label {
    color: var(--text-primary) !important;
}

/* ── Chatbot text ── */
#chatbot .message {
    color: var(--chat-text-primary) !important;
}

#chatbot .message.user { 
    color: var(--chat-text-primary) !important;
}

#chatbot .message.bot { 
    color: var(--chat-text-primary) !important;
}

/* ── Headers and main text ── */
#header-container h2, 
#header-container h3,
#tree-header h3 {
    color: var(--text-primary) !important;
}

.sidebar h4, .sidebar h3 {
    color: var(--text-primary) !important;
}

/* ── Markdown text ── */
.markdown {
    color: var(--chat-text-primary) !important;
}

.markdown h1, .markdown h2, .markdown h3, .markdown h4 {
    color: var(--chat-text-primary) !important;
}

.markdown p {
    color: var(--chat-text-primary) !important;
}


#issue-report-btn {
    background: linear-gradient(135deg, var(--accent-cyan), var(--accent-purple)) !important;
    border: none !important;
    color: white !important;
    border-radius: 12px !important;
    padding: 10px 16px !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    cursor: pointer !important;
    box-shadow: var(--accent-glow-cyan) !important;
    transition: all 0.3s ease !important;
    margin-left: auto !important;
}

#issue-report-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 0 25px rgba(0, 217, 255, 0.5) !important;
}

body.light-mode #theme-toggle-btn {
    background: linear-gradient(135deg, #0277bd, #1565c0) !important;
    box-shadow: 0 0 20px rgba(2, 119, 189, 0.25) !important;
}

body.light-mode #issue-report-btn {
    background: linear-gradient(135deg, rgba(30, 144, 255, 0.15), rgba(255, 152, 0, 0.15)) !important;
    border-color: #ff9800 !important;
    color: #0277bd !important;
}

body.light-mode #issue-report-btn:hover {
    background: linear-gradient(135deg, rgba(30, 144, 255, 0.3), rgba(255, 152, 0, 0.3)) !important;
}

#issue-modal-overlay[style*="display: block"] {
    display: flex !important;
    align-items: center;
    background: rgba(0,0,0,0.6);
    backdrop-filter: blur(6px);
}

#issue-modal {
    background: var(--background-primary);
    border-radius: 16px;
    padding: 24px;
    width: 420px;
    box-shadow: 0 20px 60px rgba(0,0,0,0.5);
    animation: fadeInScale 0.2s ease;
}

@keyframes fadeInScale {
    from {
        opacity: 0;
        transform: scale(0.95);
    }
    to {
        opacity: 1;
        transform: scale(1);
    }
}
"""

def build_walkthrough_js(steps: list) -> str:  
    steps_json = json.dumps(steps)  
    return f"""  
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap" rel="stylesheet">  

<script>
(function() {{
    const STEPS = {steps_json};
    const STORAGE_KEY = 'datasetchatty_tour_done';
    let currentStep = -1;

    // ── LIGHT MODE FIX ──
    function enableLightMode() {{
        if (!document.body.classList.contains('light-mode')) {{
            document.body.classList.add('light-mode');
            console.log('Light mode applied');
        }}
    }}

    // Warte, bis Gradio vollständig gerendert ist
    function waitForGradioBody(retries = 20, interval = 100) {{
        let attempts = 0;
        const timer = setInterval(() => {{
            attempts++;
            if (document.body) {{
                enableLightMode();
                clearInterval(timer);
            }} else if (attempts >= retries) {{
                clearInterval(timer);
            }}
        }}, interval);
    }}

    if (document.readyState === 'loading') {{
        document.addEventListener('DOMContentLoaded', waitForGradioBody);
    }} else {{
        waitForGradioBody();
    }}

  function $(sel) {{ return document.querySelector(sel); }}  
  function getTooltip() {{ return $('#walkthrough-tooltip'); }}  
  function getOverlay()  {{ return $('#walkthrough-overlay'); }}  
  function getFroggy()   {{ return $('#froggy-guide'); }}  
  
   /* ── Sidebar ── */
  function isSidebarOpen() {{
    const panel = document.querySelector('.sidebar.right');
    return panel ? panel.classList.contains('open') : false;
  }}

  function openSidebar() {{
    return new Promise((resolve) => {{
      if (isSidebarOpen()) {{ resolve(); return; }}
      const toggle = document.querySelector('button[aria-label="Toggle Sidebar"]');
      if (!toggle) {{ resolve(); return; }}
      toggle.click();
      let attempts = 0;
      const poll = setInterval(() => {{
        attempts++;
        if (isSidebarOpen() || attempts > 20) {{
          clearInterval(poll);
          setTimeout(resolve, 150);
        }}
      }}, 50);
    }});
  }}

  /* ── Tooltip positioning ── */
  function positionTooltip(rect, side) {{
    const tip = getTooltip();
    const tw  = tip.offsetWidth  || 340;
    const th  = tip.offsetHeight || 240;
    const pad = 24;
    const vw  = window.innerWidth;
    const vh  = window.innerHeight;
    let top, left;

    if (!rect || side === 'center') {{
      top  = (vh - th) / 2;
      left = (vw - tw) / 2;
    }} else {{
      switch (side) {{
        case 'left':
          top  = rect.top;
          left = rect.left - tw - pad;
          break;
        case 'right':
          top  = rect.top;
          left = rect.right + pad;
          break;
        case 'top':
          top  = rect.top - th - pad;
          left = rect.left + (rect.width - tw) / 2;
          break;
        case 'bottom':
        default:
          top  = rect.bottom + pad;
          left = rect.left + (rect.width - tw) / 2;
          break;
      }}
    }}

    top  = Math.max(pad, Math.min(top,  vh - th - pad));
    left = Math.max(pad, Math.min(left, vw - tw - pad));
    tip.style.top       = top  + 'px';
    tip.style.left      = left + 'px';
    tip.style.transform = 'none';
  }}

  /* ── Froggy hops to the border of the spotlight rect ── */  
    function moveFroggy(rect) {{  
        const frog = getFroggy();  
        
        if (!rect) {{ 
            const top  = window.scrollY + 300;
            const left = window.scrollX + 750;
            frog.style.top  = top  + 'px';  
            frog.style.left = left + 'px';  
        }} else {{ 
            const top  = window.scrollY + rect.top - 60;
            const left = window.scrollX + rect.left - 60;
            frog.style.top  = top  + 'px';  
            frog.style.left = left + 'px';  
        }} 
        
        frog.classList.remove('hopping');  
        void frog.offsetWidth;
        frog.classList.add('visible', 'hopping');  
    }}

  /* ── Spotlight: clip-path hole + optional interactive pass-through ── */
  function setSpotlight(rect, interactive) {{
    const overlay = getOverlay();

    let pt = $('#wt-passthrough');
    if (!pt) {{
      pt = document.createElement('div');
      pt.id = 'wt-passthrough';
      pt.style.cssText = `
        position:fixed; z-index:9999; background:transparent;
        border-radius:20px; border:2px dashed rgba(0,217,255,0.6);
        pointer-events:none; transition:all 0.35s ease;
        box-shadow:0 0 30px rgba(0,217,255,0.3); display:none;
      `;
      document.body.appendChild(pt);
    }}

    if (!rect) {{
      overlay.style.clipPath      = 'none';
      overlay.style.pointerEvents = 'auto';
      pt.style.display            = 'none';
      return;
    }}

    const pad = 16;
    const x   = rect.left   - pad;
    const y   = rect.top    - pad;
    const w   = rect.width  + pad * 2;
    const h   = rect.height + pad * 2;
    const vw  = window.innerWidth;
    const vh  = window.innerHeight;

    overlay.style.clipPath = `polygon(
      0 0, ${{vw}}px 0, ${{vw}}px ${{vh}}px, 0 ${{vh}}px,
      0 ${{y}}px, ${{x}}px ${{y}}px,
      ${{x}}px ${{y + h}}px, ${{x + w}}px ${{y + h}}px,
      ${{x + w}}px ${{y}}px, 0 ${{y}}px
    )`;

    if (interactive) {{
      overlay.style.pointerEvents = 'none';
      pt.style.display = 'block';
      pt.style.top     = y + 'px';
      pt.style.left    = x + 'px';
      pt.style.width   = w + 'px';
      pt.style.height  = h + 'px';
    }} else {{
      overlay.style.pointerEvents = 'auto';
      pt.style.display = 'none';
    }}
  }}

  function clearSpotlight() {{
    const pt = $('#wt-passthrough');
    if (pt) pt.style.display = 'none';
    const overlay = getOverlay();
    overlay.style.clipPath      = 'none';
    overlay.style.pointerEvents = 'auto';
  }}

  function refreshDots() {{
    document.querySelectorAll('.step-dot')
      .forEach((d, i) => d.classList.toggle('active', i <= currentStep));
  }}

  function waitForElement(selector, retries = 20, interval = 80) {{
    return new Promise((resolve) => {{
      let n = 0;
      const poll = setInterval(() => {{
        const el = document.querySelector(selector);
        n++;
        if (el || n >= retries) {{ clearInterval(poll); resolve(el || null); }}
      }}, interval);
    }});
  }}

  function resizeGraph() {{
        const container = document.getElementById("tree-panel");
        if (!container) return;

        const width = container.clientWidth;
        const height = container.clientHeight;

        console.log("Resize graph:", width, height);

        if (typeof drawKnowledgeGraph === "function") {{
            drawKnowledgeGraph(width, height);
        }}
    }}

  /* ── Main step renderer ── */
  async function showStep(n) {{  
    const overlay = getOverlay();  
    const tip     = getTooltip();  
    const frog    = getFroggy();  
  
    document.querySelectorAll('.wt-spotlight')  
        .forEach(el => el.classList.remove('wt-spotlight'));  
    clearSpotlight();  
  
    if (n < 0 || n >= STEPS.length) {{  
        overlay.style.display = 'none';  
        overlay.classList.remove('active');  
        tip.classList.remove('active');  
        frog.classList.remove('visible', 'hopping');  
        currentStep = -1;  
        return;  
    }} 

    currentStep = n;
    const step  = STEPS[n];

    tip.querySelector('h3').textContent       = step.title;
    tip.querySelector('p').textContent        = step.content;
    tip.querySelector('.next-btn').textContent =
      n === STEPS.length - 1 ? 'Finish 🎉' : 'Next →';

    refreshDots();
    overlay.classList.add('active');
    tip.classList.add('active');

    if (!step.selector) {{
      setSpotlight(null, false);
      positionTooltip(null, 'center');
      moveFroggy(null);
      return;
    }}

    if (step.openSidebar) {{
      await openSidebar();
    }}

    const el = await waitForElement(step.selector);

    if (!el) {{
      console.warn('[Walkthrough] Element not found:', step.selector);
      setSpotlight(null, false);
      positionTooltip(null, 'center');
      moveFroggy(null);
      return;
    }}

    el.scrollIntoView({{ behavior: 'smooth', block: 'nearest' }});
    await new Promise(r => setTimeout(r, 160));

    const base = el.getBoundingClientRect();

    const ex = step.expandSpotlight || {{}};
    const expandedRect = {{
      top:    base.top    - (ex.top    || 0),
      left:   base.left   - (ex.left   || 0),
      width:  base.width  + (ex.left   || 0) + (ex.right  || 0),
      height: base.height + (ex.top    || 0) + (ex.bottom || 0),
      right:  base.right  + (ex.right  || 0),
      bottom: base.bottom + (ex.bottom || 0),
    }};

    setSpotlight(expandedRect, !!step.interactive);
    positionTooltip(expandedRect, step.position);
    moveFroggy(expandedRect);
  }}

  function showTokenToast() {{
        const toast = document.getElementById('token-toast');
        if (!toast) return;
        toast.classList.add('show');
        setTimeout(() => toast.classList.remove('show'), 4000);
    }}

    function bindTokenCheck() {{
        const sendBtn = document.getElementById('send-btn');
        const msgInput = document.querySelector('#msg-input textarea');
        const tokenStatus = document.querySelector('#token-status');

        if (!sendBtn || !msgInput || !tokenStatus) return;

        sendBtn.addEventListener('click', () => {{
            const tokenSet = tokenStatus.dataset.tokenSet === "true";
            if (!tokenSet && msgInput.value.trim()) {{
                showTokenToast();
            }}
        }});

        msgInput.addEventListener('keypress', (e) => {{
            if (e.key === 'Enter') {{
                const tokenSet = tokenStatus.dataset.tokenSet === "true";
                if (!tokenSet && msgInput.value.trim()) {{
                    e.preventDefault();  // optional, falls Enter nicht direkt submitten soll
                    showTokenToast();
                }}
            }}
        }});
    }}

    function waitForElements(retries = 20, interval = 100) {{
        let attempts = 0;
        const timer = setInterval(() => {{
            attempts++;
            const sendBtn = document.getElementById('send-btn');
            const msgInput = document.querySelector('#msg-input textarea');
            const tokenStatus = document.querySelector('#token-status');
            if (sendBtn && msgInput && tokenStatus) {{
                clearInterval(timer);
                bindTokenCheck();
            }} else if (attempts >= retries) {{
                clearInterval(timer);
                console.warn("[TokenToast] Elemente nicht gefunden");
            }}
        }}, interval);
    }}

    if (document.readyState === 'loading') {{
        document.addEventListener('DOMContentLoaded', waitForElements);
    }} else {{
        waitForElements();
    }}

    // Direkt beim Demo-Load starten
    window.addEventListener('load', () => {{
        bindTokenToast();
    }});

    // ── ISSUE REPORT FUNCTIONS ──
    window.openIssueReport = function() {{
        const modal = document.getElementById('issue-report-modal');
        if (modal) modal.classList.add('active');
        const inputs = modal.querySelectorAll('input, textarea, select');
        inputs.forEach(el=>el.value = el.defaultValue||'');
        const success = modal.querySelector('#issue-success-message');
        if (success) success.classList.remove('show');
    }};
    window.closeIssueReport = function() {{
        const modal = document.getElementById('issue-report-modal');
        if (modal) modal.classList.remove('active');
    }};
    document.addEventListener('DOMContentLoaded', () => {{
        const gradioBtn = document.querySelector('#issue-submit-btn');
        if (!gradioBtn) return;
        gradioBtn.addEventListener('click', () => {{
            const successMsg = document.getElementById('issue-success-message');
            if (successMsg) {{
                successMsg.classList.add('show');
                setTimeout(() => {{
                    successMsg.classList.remove('show');
                    window.closeIssueReport();
                }}, 2000);
            }}
        }});
    }});

    window.submitIssueReportForm = function() {{
        const cat   = document.getElementById('issue-category');
        const title = document.getElementById('issue-title');
        const desc  = document.getElementById('issue-description');
        const sev   = document.getElementById('issue-severity');

        // Validierung
        if (!cat.value || !title.value.trim() || !desc.value.trim()) {{
            alert('Please fill in all required fields.');
            return;
        }}

        // Werte in Gradio-Komponenten schreiben
        function setGradioInput(selector, value) {{
            const el = document.querySelector(selector);
            if (!el) return;
            const input = el.querySelector('input, textarea');
            if (input) {{
                const nativeSetter = Object.getOwnPropertyDescriptor(
                    window.HTMLInputElement.prototype, 'value'
                ) || Object.getOwnPropertyDescriptor(
                    window.HTMLTextAreaElement.prototype, 'value'
                );
                nativeSetter.set.call(input, value);
                input.dispatchEvent(new Event('input', {{ bubbles: true }}));
            }}
        }}

        function setGradioDropdown(labelText, value) {{
            // Gradio Dropdown: Suche die richtige Komponente über elem_id
            // und simuliere Auswahl über das interne Svelte-Event-System
            const containers = document.querySelectorAll('.hidden-issue-inputs select, #hidden-issue-inputs select');
            containers.forEach(sel => {{
                if (sel.closest('[id*="gradio-issue-category"]') || 
                    sel.closest('[id*="gradio-issue-severity"]')) {{
                    sel.value = value;
                    sel.dispatchEvent(new Event('change', {{ bubbles: true }}));
                }}
            }});
        }}

        setGradioInput('#gradio-issue-title input, [id*="gradio-issue-title"] input', title.value.trim());
        setGradioInput('[id*="gradio-issue-description"] textarea', desc.value.trim());

        // Für Dropdowns: Gradio nutzt eigene Komponenten – direkter Trigger über Gradio API
        // Einfachste zuverlässige Methode: hidden Inputs als Textbox statt Dropdown
        const gradioTitle = document.querySelector('[id*="gradio-issue-title"] input, [id*="gradio-issue-title"] textarea');
        const gradioDesc  = document.querySelector('[id*="gradio-issue-description"] textarea, [id*="gradio-issue-description"] input');
        const gradioBtn   = document.getElementById('issue-submit-btn-hidden');

        if (gradioTitle) {{
            const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value') ||
                        Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value');
            setter.set.call(gradioTitle, title.value.trim());
            gradioTitle.dispatchEvent(new Event('input', {{ bubbles: true }}));
        }}
        if (gradioDesc) {{
            const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value');
            setter.set.call(gradioDesc, desc.value.trim());
            gradioDesc.dispatchEvent(new Event('input', {{ bubbles: true }}));
        }}

        if (gradioBtn) gradioBtn.click();

        // Erfolg anzeigen
        const success = document.getElementById('issue-success-message');
        if (success) {{
            success.classList.add('show');
            setTimeout(() => {{
                success.classList.remove('show');
                window.closeIssueReport();
            }}, 2000);
        }} else {{
            window.closeIssueReport();
        }}
    }};
  
  /* ── Public API ── */  
  window.startWalkthrough = () => {{  
    localStorage.removeItem(STORAGE_KEY);  
    currentStep = -1;  
    showStep(0);  
  }};  
  window.nextStep         = () => showStep(currentStep + 1);  
  window.skipWalkthrough  = () => {{  
    localStorage.setItem(STORAGE_KEY, '1');  
    showStep(-1);  
  }};  
  
  /* Auto-launch once per browser */  
  function autoLaunch() {{  
    if (!localStorage.getItem(STORAGE_KEY)) {{  
      setTimeout(() => showStep(0), 1200);  
    }}  
  }}  
  
  if (document.readyState === 'loading') {{  
    document.addEventListener('DOMContentLoaded', autoLaunch);  
  }} else {{  
    autoLaunch();  
  }}  
}})();  
</script>  
"""


# ============================================================
# GRADIO INTERFACE
# ============================================================

def create_interface():
    with gr.Blocks(
        css=CSS_DARK,
        title="DatasetChatty",
        head=build_walkthrough_js(WALKTHROUGH_STEPS)
    ) as demo:
        # ── Walkthrough ──
        gr.HTML(f"""
        <div id="walkthrough-overlay"></div>
        <div id="froggy-guide">🐸</div>
        <div id="token-toast">
            <span>⚠️ API Token erforderlich - Bitte in der Sidebar setzen!</span>
            <button id="token-toast-close" onclick="this.parentElement.classList.remove('show')">✕</button>
        </div>
        <div id="walkthrough-tooltip">
            <div class="step-indicator">
                {''.join('<div class="step-dot"></div>' for _ in WALKTHROUGH_STEPS)}
            </div>
            <h3>Welcome!</h3>
            <p>Content here</p>
            <div class="wt-buttons">
                <button class="skip-btn" onclick="skipWalkthrough()">Skip tour</button>
                <button class="next-btn" onclick="nextStep()">Next →</button>
            </div>
        </div>
        """)

        # ── State ──
        session_id_state = gr.State(str(uuid.uuid4()))
        api_token_state = gr.State("")
        history_state    = gr.State([])
        theme_state      = gr.State("dark")
        dummy_bool_state = gr.State(False)

        with gr.Column(visible=False, elem_id="issue-modal-overlay") as issue_modal:
            with gr.Column(elem_id="issue-modal"):

                gr.Markdown("## 📋 Report an Issue")
                gr.Markdown("Help us improve! Report bugs, suggest features, or share feedback.")

                cat_input = gr.Dropdown(
                    label="Category *",
                    choices=["🐛 Bug", "💡Feature Request", "🔍 Search Quality", "⚡Performance", "🎨 UI/UX", "📝 Other"]
                )

                title_input = gr.Textbox(
                    label="Title *",
                    placeholder="Brief summary...",
                    max_lines=1
                )

                desc_input = gr.Textbox(
                    label="Description *",
                    placeholder="Provide details...",
                    lines=4
                )

                sev_input = gr.Dropdown(
                    label="Severity",
                    choices=["low","medium","high","critical"],
                    value="medium"
                )

                issue_status = gr.HTML("")

                with gr.Row():
                    cancel_btn = gr.Button("Cancel")
                    issue_submit_btn = gr.Button("Submit Report", variant="primary")

        def submit_and_close(cat, title, desc, sev, session_id):
            status, flag = submit_issue_report(cat, title, desc, sev, session_id)
            return status, flag, gr.update(visible=False)
        
        issue_submit_btn.click(
            submit_and_close,
            inputs=[cat_input, title_input, desc_input, sev_input, session_id_state],
            outputs=[issue_status, dummy_bool_state, issue_modal]
        )

        # ── Header ──
        with gr.Row(elem_id="header-container"):
            froggy_btn = gr.Button("🐸", elem_id="froggy-btn", scale=0)
            with gr.Column(scale=1):
                gr.Markdown("""
                # DatasetChat
                <span style="color:var(--text-secondary);font-size:14px;font-weight:500;">AI-powered research assistant for Hugging Face Hub</span>
                """)
            issue_btn = gr.Button("📋 Report Issue", elem_id="issue-report-btn", scale=0)
            theme_toggle = gr.Button("🌙 Dark Mode", elem_id="theme-toggle-btn", scale=0)

        # ── Main content ──
        with gr.Column(elem_id="main-container"):

            with gr.Column(scale=4, elem_id="tree-panel-container"):
                gr.HTML('''<div id="tree-header" style="position: relative;">
                    <h3 style="margin: 0;">🧠 Knowledge Graph</h3>
                    <button id="tree-fullscreen-btn" 
                        style="position: absolute; right: 20px; top: 50%; transform: translateY(-50%);
                            background: rgba(0, 217, 255, 0.1); border: 1px solid rgba(0, 217, 255, 0.3);
                            padding: 8px 12px; border-radius: 8px; cursor: pointer; font-size: 16px;
                            color: var(--text-primary); transition: all 0.3s;">
                        🖥️
                    </button>
                    <button id="tree-close-btn" 
                        style="display: none; position: absolute; right: 20px; top: 50%; transform: translateY(-50%);
                            background: rgba(255, 68, 68, 0.1); border: 1px solid rgba(255, 68, 68, 0.3);
                            padding: 8px 12px; border-radius: 8px; cursor: pointer; font-size: 16px;
                            color: #fca5a5; transition: all 0.3s;">
                        ✖️
                    </button>
                </div>''')
                tree_panel = gr.HTML(
                    value=LogTreeTracker().render_tree_svg("dark"),
                    elem_id="tree-panel"
                )

            with gr.Column(scale=8, elem_id="chat-panel"):
                chatbot = gr.Chatbot(elem_id="chatbot", show_label=False, height=480)
                with gr.Row(elem_id="input-container"):
                    msg = gr.Textbox(
                        placeholder="Ask about datasets, papers, or models...",
                        show_label=False, elem_id="msg-input", scale=5
                    )
                    send_btn = gr.Button("Send", elem_id="send-btn", scale=1)
                    new_chat_btn = gr.Button("🔄 New Chat", elem_id="new-chat-btn", scale=1)

        # ── Sidebar ──
        with gr.Sidebar(position="right"):
            gr.Markdown("### ⚙️ Settings")

            with gr.Group():
                gr.HTML("<h4 style='margin-top:0;color:var(--text-primary);'>🔑 API Token</h4>")
                api_input = gr.Textbox(
                    label="Hugging Face Token", type="password",
                    placeholder="hf_...", show_label=False
                )
                save_btn = gr.Button("Save Token", variant="primary", size="sm")
                token_status = gr.HTML(
                    #'<div id="token-status" class="pending">Not configured</div>'
                    
                    '<div id="token-status" class="pending" data-token-set="false">Not configured</div>'
                    
                )

            gr.Markdown("---")

            with gr.Group(elem_id="stats-section"):
                gr.Markdown("### Session")
                session_display = gr.Textbox(label="Session ID", interactive=False)
                query_count     = gr.Number(label="Queries", value=0, interactive=False)

            with gr.Accordion("🔒 Admin Panel", open=False):
                admin_input = gr.Textbox(label="Admin Key", type="password")
                
                admin_btn = gr.Button("Load Logs")
                admin_output = gr.Dataframe(headers=["Timestamp","Session ID","Query","Response"])
                admin_btn.click(get_all_queries, inputs=admin_input, outputs=admin_output)

                download_btn = gr.File(label="Download Logs")  

                admin_download_btn = gr.Button("Download CSV")
                admin_download_btn.click(download_logs, inputs=admin_input, outputs=download_btn)

            # Issue Reports
            with gr.Accordion("📋 Issue Reports", open=False):
                issue_stats = gr.HTML(value="Loading...")
                load_stats_btn = gr.Button("Load Statistics", size="sm")
                load_stats_btn.click(get_issue_stats, inputs=admin_input, outputs=issue_stats)
                issue_reports_df = gr.Dataframe(label="Reports")
                load_reports_btn = gr.Button("Load Reports", size="sm")
                load_reports_btn.click(get_issue_reports_admin, inputs=admin_input, outputs=issue_reports_df)
                issue_file = gr.File(label="CSV Export")
                download_btn_issues = gr.Button("Export CSV", size="sm")
                download_btn_issues.click(download_issue_reports, inputs=admin_input, outputs=issue_file)

        # ── Event Bindings ──
        froggy_btn.click(fn=None, js="() => window.startWalkthrough()")

        issue_btn.click(
            fn=lambda: gr.update(visible=True),
            outputs=issue_modal
        )

        cancel_btn.click(
            fn=lambda: gr.update(visible=False),
            outputs=issue_modal
        )
        # ── Events ──

        def toggle_theme(current_theme):
            new_theme = "light" if current_theme == "dark" else "dark"
            btn_text = "☀️ Light Mode" if new_theme == "dark" else "🌙 Dark Mode"
            return new_theme, btn_text

        theme_toggle.click(
            toggle_theme,
            inputs=theme_state,
            outputs=[theme_state, theme_toggle],
            js="() => { const body = document.body; body.classList.toggle('light-mode'); }"
        )

        def save_token(token, session_id):
            if session_manager.validate_token(token):
                session_manager.set_token(session_id, token)
                return '<div id="token-status" class="valid" data-token-set="true">✓ Token valid</div>', token
            return '<div id="token-status" class="invalid" data-token-set="false">✗ Invalid token</div>', ""

        save_btn.click(save_token, inputs=[api_input, session_id_state], outputs=[token_status, api_token_state])

        demo.load(lambda: str(uuid.uuid4()), outputs=session_display)

        # Fullscreen Mode für Knowledge Graph
        demo.load(
            None,
            None,
            None,
            js="""
            () => {
                const panel = document.getElementById('tree-panel-container');
                const openBtn = document.getElementById('tree-fullscreen-btn');
                const closeBtn = document.getElementById('tree-close-btn');

                if (!panel || !openBtn || !closeBtn) {
                    console.warn('[Fullscreen] Elements not found');
                    return;
                }

                openBtn.onclick = () => {
                    panel.classList.add('fullscreen');

                    setTimeout(() => {
                        resizeGraph();   
                    }, 150);
                };

                closeBtn.onclick = () => {
                    panel.classList.remove('fullscreen');

                    setTimeout(() => {
                        resizeGraph();   
                    }, 150);
                };

                // Escape key to close fullscreen
                document.addEventListener('keydown', (e) => {
                    if (e.key === 'Escape' && panel.classList.contains('fullscreen')) {
                        panel.classList.remove('fullscreen');
                        console.log('[Fullscreen] EXIT (ESC)');
                        setTimeout(() => {
                            window.dispatchEvent(new Event('resize'));
                        }, 100);
                    }
                });
            }
            """
        )


        def handle_submit(user_msg, history, session_id, token, theme):
            if not user_msg.strip():
                return history, history, "", LogTreeTracker().render_tree_svg(theme), 0

            if not token:
                error = "⚠️ Please set your API token in the sidebar first!"
                new_history = history + [
                    {"role": "user",      "content": user_msg},
                    {"role": "assistant", "content": error},
                ]
                return new_history, new_history, "", LogTreeTracker().render_tree_svg(theme), len(new_history)

            history = history + [{"role": "user", "content": user_msg}]
            final_answer = ""
            current_tree = LogTreeTracker().render_tree_svg(theme)

            for tree_html, answer in ask_llm_streaming(user_msg, history[:-1], token, theme):
                current_tree = tree_html
                if answer:
                    final_answer = answer
                yield (
                    history + ([{"role": "assistant", "content": final_answer}] if final_answer else []),
                    history + ([{"role": "assistant", "content": final_answer}] if final_answer else []),
                    "",
                    current_tree,
                    len(history) // 2
                )

            if final_answer:
                updated_history = history + [{"role": "assistant", "content": final_answer}]
                qid = query_logger.log_query(session_id, user_msg)
                query_logger.update_query(qid, final_answer)
            else:
                updated_history = history
            yield updated_history, updated_history, "", current_tree, len(updated_history) // 2

        for trigger in (send_btn.click, msg.submit):
            trigger(
                handle_submit,
                inputs=[msg, history_state, session_id_state, api_token_state, theme_state],
                outputs=[chatbot, history_state, msg, tree_panel, query_count],
            )

        def new_chat(theme):
            return [], [], "", LogTreeTracker().render_tree_svg(theme), 0

        # Toast JavaScript Logic
        demo.load(
            None, None, None,
            js="""
            () => {
                function showTokenToast() {
                    const toast = document.getElementById('token-toast');
                    if (toast) {
                        toast.classList.add('show');
                        setTimeout(() => toast.classList.remove('show'), 4000);
                    }
                }
                
                document.addEventListener('DOMContentLoaded', () => {
                    const sendBtn = document.getElementById('send-btn');
                    const msgInput = document.querySelector('#msg-input textarea');
                    
                    if (sendBtn && msgInput) {
                        sendBtn.addEventListener('click', () => {
                            const tokenSet = document.querySelector('#token-status')?.dataset.tokenSet === "true";
                            if (!tokenSet && msgInput.value.trim()) {
                                showTokenToast();
                            }
                        });
                        
                        msgInput.addEventListener('keypress', (e) => {
                            if (e.key === 'Enter' && e.ctrlKey) {
                                const tokenSet = document.querySelector('#token-status')?.dataset.tokenSet === "true";
                                if (!tokenSet && msgInput.value.trim()) {
                                    showTokenToast();
                                }
                            }
                        });
                     }
                 });
            }
            """
        )

        new_chat_btn.click(
            new_chat,
            inputs=theme_state,
            outputs=[chatbot, history_state, msg, tree_panel, query_count],
        )

    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(debug=True)