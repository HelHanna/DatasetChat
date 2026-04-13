import json
import logging
import sys
from typing import Set, List, Dict, Optional, Tuple, Any, Callable
import time
from collections import defaultdict
import re

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
        "filter":   "⚖️",
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

    def render_tree_svg(self) -> str:
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

        svg = f'''
        <div style="overflow: auto; width: 100%; height: 100%;">
            <svg width="{svg_width}" height="{svg_height}" 
                xmlns="http://www.w3.org/2000/svg"
                style="background: linear-gradient(135deg, #f8fafc 0%, #f0f4f8 100%);
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

                        svg += f'<path d="{path}" stroke="#6366f1" stroke-width="2" fill="none" opacity="0.4"/>'
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


def parse_paper_from_mcp_result(mcp_text: str) -> List[Dict[str, str]]:
    """Parse paper results from MCP search"""
    papers = []
    for section in mcp_text.split('\n---\n'):
        if not section.strip() or 'papers matched the query' in section:
            continue
        pd = {"paper_id": "", "title": "", "published": "", "authors": "",
              "abstract": "", "ai_summary": "", "ai_keywords": "", "link": "", "full_text": section}
        m = re.search(r'https://hf\.co/papers/([\d.]+)', section)
        if m:
            pd["paper_id"] = m.group(1)
            pd["link"] = m.group(0)
        m = re.search(r'## (.+?)$', section, re.MULTILINE)
        if m:
            pd["title"] = m.group(1).strip()
        m = re.search(r'Published on (.+?)$', section, re.MULTILINE)
        if m:
            pd["published"] = m.group(1).strip()
        m = re.search(r'\*\*Authors:\*\* (.+?)$', section, re.MULTILINE)
        if m:
            pd["authors"] = m.group(1).strip()
        m = re.search(r'### Abstract\s*\n\n(.+?)(?=\n\n\*\*|$)', section, re.DOTALL)
        if m:
            pd["abstract"] = m.group(1).strip()
        m = re.search(r'### AI Generated Summary\s*\n\n(.+?)(?=\n\n\*\*|$)', section, re.DOTALL)
        if m:
            pd["ai_summary"] = m.group(1).strip()
        m = re.search(r'\*\*AI Keywords\*\*: (.+?)$', section, re.MULTILINE)
        if m:
            pd["ai_keywords"] = m.group(1).strip()
        if pd["paper_id"]:
            papers.append(pd)
    return papers

# ============================================================
# RESPONSE FORMATS FOR LLM JSON SCHEMA
# ============================================================

response_format = {
    "type": "json_schema",
    "json_schema": {
        "name": "dataset_scores",
        "schema": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "relevant": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1}
                },
                "required": ["name", "relevant", "confidence"],
                "additionalProperties": False
            }
        }
    }
}

judgement_response_format = {
    "type": "json_schema",
    "json_schema": {
        "name": "dataset_judgements",
        "schema": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "relevant": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["name", "relevant", "reason"],
                "additionalProperties": False
            }
        },
    },
}

dataset_rerank_response_format = {
    "type": "json_schema",
    "json_schema": {
        "name": "dataset_rerank",
        "schema": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "rank": {"type": "integer"},
                    "reason": {"type": "string"},
                    "downloads": {"type": "integer"},
                    "likes": {"type": "integer"},
                    "semantic_confidence": {"type": "number"}
                },
                "required": ["name", "rank", "reason", "downloads", "likes", "semantic_confidence"],
                "additionalProperties": False
            }
        }
    }
}

paper_judgement_response_format = {
    "type": "json_schema",
    "json_schema": {
        "name": "paper_judgements",
        "schema": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "paper_id": {"type": "string"},
                    "relevant": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "reason": {"type": "string"}
                },
                "required": ["paper_id", "relevant", "confidence", "reason"],
                "additionalProperties": False
            }
        }
    }
}

tag_response_format = {
    "type": "json_schema",
    "json_schema": {
        "name": "selected_tags",
        "schema": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "relevant": {"type": "string"},
                },
                "required": ["name", "relevant"],
                "additionalProperties": False
            }
        }
    }
}

# ============================================================
# LLM AGENTS
# ============================================================

def query_understanding_agent(user_message, history, llm, tools):
    """Understand user query and select appropriate tools"""
    system = ("You are a Query Understanding and Tool Selection Agent. "
              "Use hub_repo_search for datasets/papers. "
              "Use hf_doc_search/hf_doc_fetch for coding questions. "
              "NEVER use hub_repo_details with repo_type='paper'.")
    
    messages = [{"role": "system", "content": system}] + history + [{"role": "user", "content": user_message}]
    
    return llm.chat.completions.create(
        messages=messages, 
        tools=tools, 
        tool_choice="auto",
        max_tokens=1200, 
        stream=True
    )


def semantic_paper_search_agent(user_message, history, llm, tools, mcp_client,
                                tracker: LogTreeTracker = None):
    """Search for relevant papers using semantic paper search"""
    if tracker:
        tracker.log("paper", "Running semantic paper search…")
    
    prompt = ("Research assistant. Find relevant papers using paper_search tool only. "
              "Do NOT answer directly.")
    
    messages = [{"role": "system", "content": prompt}] + history + [{"role": "user", "content": user_message}]
    
    stream = llm.chat.completions.create(
        messages=messages, 
        tools=tools,
        tool_choice="auto", 
        max_tokens=1000, 
        stream=True
    )
    
    paper_tool_calls = []
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta.tool_calls:
            for tc in delta.tool_calls:
                if "paper_search" in tc.function.name:
                    try:
                        args = json.loads(tc.function.arguments)
                        paper_tool_calls.append(args)
                    except Exception:
                        continue
    
    paper_ids, paper_data = [], {}
    for idx, args in enumerate(paper_tool_calls):
        query = args.get("query")
        if not query:
            continue
        if tracker:
            tracker.log("paper", f"Searching papers: {query}")
        try:
            result = mcp_client.call_tool("paper_search",
                                          {"query": query,
                                           "results_limit": args.get("results_limit", 10),
                                           "concise_only": args.get("concise_only", False)})
            if result:
                for item in result.get("content", []):
                    if item.get("type") == "text":
                        parsed = parse_paper_from_mcp_result(item.get("text", ""))
                        for paper in parsed:
                            pid = paper["paper_id"]
                            if pid and pid not in paper_data:
                                paper_ids.append(pid)
                                paper_data[pid] = paper
                                if tracker:
                                    tracker.log("paper", f"Found paper: {paper.get('title', pid)[:60]}")
        except Exception as e:
            logger.error(f"Paper search failed: {e}")
    
    return paper_ids, paper_data


def tag_selection_agent(user_query, dataset_id, candidate_tags, llm, max_tags=3, 
                       base_max_tokens=400, max_retries=2, tracker: LogTreeTracker = None):
    """Select relevant tags for a dataset based on user query"""
    if not candidate_tags:
        return []
    
    prompt = f"""You are filtering dataset tags for relevance.
User query: "{user_query}"
Dataset: {dataset_id}
Candidate tags: {', '.join(candidate_tags)}
Task: For each tag decide if relevant. Return JSON array with "name" and "relevant" keys.
Include at most {max_tags} tags marked as "yes".
"""
    max_tokens = base_max_tokens
    for attempt in range(max_retries + 1):
        try:
            response = llm.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens, 
                temperature=0.0, 
                response_format=tag_response_format
            )
            raw = response.choices[0].message.content
            selected = json.loads(raw)
            if isinstance(selected, dict):
                selected = [selected]
            selected = [t for t in selected if t.get("relevant", "").lower() == "yes"]
            return selected[:max_tags]
        except Exception as e:
            logger.warning(f"Tag selection failed: {e}")
            max_tokens *= 2
    return []


# Sample tags for tag matching
sample_tags = ["translation", "european", "speech", "sentence-transformers",
               "ukrainian", "russia", "conflict", "math", "reasoning"]


def tag_matching_agent(user_message, history, llm, tracker: LogTreeTracker = None) -> list:
    """Extract relevant tags from sample tags based on user query"""
    if tracker:
        tracker.log("tag", "Extracting relevant tags…")
    
    prompt = f"""Given the user query, pick up to 3 relevant tags from {sample_tags}.
Return ONLY a JSON list of strings.
Query: "{user_message}"
"""
    messages = [{"role": "system", "content": prompt}] + history + [{"role": "user", "content": user_message}]
    
    response = llm.chat.completions.create(
        messages=messages, 
        temperature=0, 
        max_tokens=200
    )
    
    raw = response.choices[0].message.content
    try:
        tags = json.loads(raw)
        if isinstance(tags, list):
            return tags[:3]
    except Exception:
        pass
    
    return []



def prefilter_judgement_agent(user_query, dataset_nodes, llm, tracker: LogTreeTracker = None,
                              batch_size=10, max_keep=15, base_max_tokens=600, max_retries=2):
    """Pre-filter datasets for relevance using cheap LLM judgement"""
    if not dataset_nodes:
        return []
    
    if tracker:
        tracker.log("filter", f"Pre-filtering {len(dataset_nodes)} candidates…")
    
    selected_ds = []
    for batch_idx, i in enumerate(range(0, len(dataset_nodes), batch_size), start=1):
        batch_nodes = dataset_nodes[i:i + batch_size]
        blocks = [f"Dataset: {ds.id}\nDescription: {ds.metadata.get('description','')}\nTags: {', '.join(ds.tags)}"
                  for ds in batch_nodes]
        prompt = f"""User query: {user_query}
For each dataset decide relevance and confidence (0-1).
Return JSON array: [{{"name":"...", "relevant":"yes/no", "confidence":0.00}}]
Datasets:
{chr(10).join(blocks)}"""
        
        attempt, max_tokens = 0, base_max_tokens
        while attempt <= max_retries:
            try:
                response = llm.chat.completions.create(
                    messages=[{"role": "system", "content": "Dataset selection agent. Output JSON only."},
                               {"role": "user", "content": prompt}],
                    max_tokens=max_tokens, 
                    temperature=0.0, 
                    response_format=response_format
                )
                raw = response.choices[0].message.content.strip()
                data = json.loads(raw)
                if isinstance(data, dict):
                    data = [data]
                
                for item in data:
                    for ds in batch_nodes:
                        if ds.id == item["name"]:
                            ds.metadata["relevant"] = item.get("relevant", "no")
                            ds.metadata["confidence"] = float(item.get("confidence", 0.0))
                
                filtered = [ds for ds in batch_nodes if ds.metadata.get("relevant") == "yes"]
                selected_ds.extend(filtered)
                break
            except Exception as e:
                err = str(e)
                if "json_validate_failed" in err or "max completion tokens" in err:
                    attempt += 1
                    max_tokens *= 2
                    continue
                break
    
    if tracker:
        tracker.log("filter", f"Pre-filter kept {len(selected_ds[:max_keep])} datasets")
    
    return selected_ds[:max_keep]


def full_context_relevance_judgement_agent(user_query, dataset_nodes, mcp_client, llm,
                                           tracker: LogTreeTracker = None, batch_size=5, max_keep=5):
    """Deep judgement of borderline datasets using full context"""
    if not dataset_nodes:
        return []
    
    if tracker:
        tracker.log("filter", f"Deep-judging {len(dataset_nodes)} borderline datasets…")
    
    selected_ds = []
    for batch_idx, i in enumerate(range(0, len(dataset_nodes), batch_size), start=1):
        batch_nodes = dataset_nodes[i:i + batch_size]
        enriched = {}
        for ds in batch_nodes:
            try:
                details = mcp_client.get_prompt("Dataset Details", {"dataset_id": ds.id})
                if details:
                    enriched[ds.id] = details[:500] + ("..." if len(details) > 500 else "")
            except Exception:
                pass
        
        blocks = [f"Dataset ID: {ds.id}\nTags: {', '.join(ds.tags)}\nDetails:\n{enriched.get(ds.id, 'No details')}"
                  for ds in batch_nodes]
        prompt = f"""Judgement Agent. User query: "{user_query}"
Datasets:
{chr(10).join(blocks)}
Return JSON array: [{{"name":"...", "relevant":"yes/no", "reason":"..."}}]"""
        
        try:
            response = llm.chat.completions.create(
                messages=[{"role": "system", "content": "Dataset judgement agent. Output JSON."},
                           {"role": "user", "content": prompt}],
                max_tokens=600, 
                temperature=0.0, 
                response_format=judgement_response_format
            )
            data = json.loads(response.choices[0].message.content.strip())
            if isinstance(data, dict):
                data = [data]
            kept_ids = [d["name"] for d in data if d.get("relevant", "").lower() == "yes"]
            filtered = [ds for ds in batch_nodes if ds.id in kept_ids]
            selected_ds.extend(filtered)
        except Exception as e:
            logger.error(f"Judgement batch {batch_idx} failed: {e}")
        
        if len(selected_ds) >= max_keep:
            break
    
    return selected_ds[:max_keep]


def paper_relevance_judgement_agent(user_query, papers, llm, tracker: LogTreeTracker = None,
                                    batch_size=5, max_content_chars=400):
    """Judge relevance of papers to user query"""
    if not papers:
        return []
    
    papers_with_content = [
        p for p in papers
        if (p.get("metadata", {}).get("abstract") or p.get("metadata", {}).get("ai_summary", "")).strip()
    ]
    
    if not papers_with_content:
        return []
    
    all_judged = []
    for batch_idx in range(0, len(papers_with_content), batch_size):
        batch = papers_with_content[batch_idx:batch_idx + batch_size]
        paper_blocks = []
        for p in batch:
            paper_id = p.get("paper_id", "unknown")
            meta = p.get("metadata", {})
            content = (meta.get("abstract") or meta.get("ai_summary", ""))[:max_content_chars]
            block = f"=== Paper {paper_id} ===\nTitle: {meta.get('title', paper_id)}\nContent: {content}\nKeywords: {meta.get('ai_keywords', '')[:100]}"
            paper_blocks.append(block)
        
        prompt = f"""Judge paper relevance for query: "{user_query}"
{"".join(paper_blocks)}
Return JSON: [{{"paper_id":"...", "relevant":"yes/no", "confidence":0.85, "reason":"..."}}]"""
        
        try:
            response = llm.chat.completions.create(
                messages=[{"role": "system", "content": "Paper relevance judge. Output JSON."},
                           {"role": "user", "content": prompt}],
                temperature=0.0, 
                max_tokens=1600, 
                response_format=paper_judgement_response_format
            )
            judged = json.loads(response.choices[0].message.content.strip())
            if isinstance(judged, dict):
                judged = [judged]
            all_judged.extend(judged)
        except Exception as e:
            logger.error(f"Paper judgement batch failed: {e}")
    
    kept = [p for p in all_judged if p.get("relevant", "").lower() == "yes" and float(p.get("confidence", 0)) >= 0.4]
    
    if tracker:
        tracker.log("filter", f"Paper judgement kept {len(kept)} relevant papers")
    
    return kept


def reranking_agent(user_query, datasets, enriched, llm, tracker: LogTreeTracker = None):
    """Re-rank datasets based on semantic relevance and metadata"""
    if tracker:
        tracker.log("rank", f"Re-ranking top {len(datasets)} candidates…")
    
    blocks = []
    for ds in datasets:
        block = f"""Dataset ID: {ds.id}
Score: {ds.metadata.get("rank_score")} | Downloads: {ds.metadata.get("downloads")}
Details: {enriched.get(ds.id, "No details")[:400]}"""
        blocks.append(block)
    
    prompt = f"""Re-ranking agent. Query: "{user_query}"
Candidates:
{chr(10).join(blocks)}
Return JSON: [{{"name":"...", "rank":1, "reason":"...", "downloads":0, "likes":0, "semantic_confidence":0.9}}]
Rank 1 = best. Semantic match first."""
    
    response = llm.chat.completions.create(
        messages=[{"role": "system", "content": "Re-ranking agent. Output JSON."},
                  {"role": "user", "content": prompt}],
        temperature=0.0, 
        max_tokens=1000, 
        response_format=dataset_rerank_response_format
    )
    
    raw = response.choices[0].message.content.strip()
    try:
        data = json.loads(raw)
    except Exception:
        data = []
    
    if isinstance(data, dict):
        data = [data]
    elif not isinstance(data, list):
        data = []
    
    return data