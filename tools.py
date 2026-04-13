# ============================================================
# TOOLS DEFINITION
# ============================================================
MCP_SERVER = "https://huggingface.co/mcp"

tools = [
    {
        "type": "mcp",
        "server_label": "hf-mcp-server",
        "server_url": MCP_SERVER,
        "function": {
            "name": "dataset_search",
            "description": "Find datasets hosted on Hugging Face.",
            "parameters": {
                "query": {"type": "string"},
                "author": {"type": "string"},
                "tags": {"type": "array", "items": {"type": "string"}},
                "sort": {
                    "type": "string",
                    "enum": ["trendingScore", "downloads", "likes", "createdAt", "lastModified"]
                },
                "limit": {"type": "integer", "minimum": 1, "maximum": 100}
            }
        },
        "require_approval": "never",
        "strict": True,
    },
    {
        "type": "mcp",
        "server_label": "hf-mcp-server",
        "server_url": MCP_SERVER,
        "function": {
            "name": "hub_repo_search",
            "description": "Unified search for repos (datasets/models) on Hugging Face",
            "parameters": {
                "query": {"type": "string"},
                "repo_types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of repo types to search, e.g., ['dataset']"
                },
                "limit": {"type": "integer", "minimum": 1, "maximum": 100}
            }
        },
        "require_approval": "never",
        "strict": True,
    },
    {
        "type": "mcp",
        "server_label": "hf-mcp-server",
        "server_url": MCP_SERVER,
        "function": {
            "name": "paper_search",
            "description": "Find ML research papers.",
            "parameters": {
                "query": {"type": "string", "minLength": 3, "maxLength": 200},
                "results_limit": {"type": "integer"},
                "concise_only": {"type": "boolean"}
            }
        },
        "require_approval": "never",
        "strict": True,
    },
    {
        "type": "mcp",
        "server_label": "hf-mcp-server",
        "server_url": MCP_SERVER,
        "function": {
            "name": "model_search",
            "description": "Find ML models hosted on Hugging Face.",
            "parameters": {
                "query": {"type": "string"},
                "author": {"type": "string"},
                "task": {"type": "string"},
                "library": {"type": "string"},
                "sort": {
                    "type": "string",
                    "enum": ["trendingScore", "downloads", "likes", "createdAt", "lastModified"]
                },
                "limit": {"type": "integer", "minimum": 1, "maximum": 100}
            }
        },
        "require_approval": "never",
        "strict": True,
    },
    {
        "type": "mcp",
        "server_label": "hf-mcp-server",
        "server_url": MCP_SERVER,
        "function": {
            "name": "hub_repo_details",
            "description": "Fetch detailed metadata about a HF repo.",
            "parameters": {
                "repo_ids": {"type": "array", "items": {"type": "string"}},
                "repo_type": {
                    "type": "string",
                    "enum": ["model", "dataset", "space"]
                },
                "include_readme": {"type": "boolean"}
            }
        },
        "require_approval": "never",
        "strict": True,
    },
    {
        "type": "mcp",
        "server_label": "hf-mcp-server",
        "server_url": MCP_SERVER,
        "function": {
            "name": "hf_doc_search",
            "description": "Search HF documentation for coding help.",
            "parameters": {
                "query": {"type": "string", "maxLength": 200},
                "product": {"type": "string"}
            }
        },
        "require_approval": "never",
        "strict": True
    },
    {
        "type": "mcp",
        "server_label": "hf-mcp-server",
        "server_url": MCP_SERVER,
        "function": {
            "name": "hf_doc_fetch",
            "description": "Fetch HF documentation page.",
            "parameters": {
                "doc_url": {"type": "string", "maxLength": 200},
                "offset": {"type": "integer", "minimum": 0}
            }
        },
        "require_approval": "never",
        "strict": True,
    },
]