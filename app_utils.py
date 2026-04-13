
import math
from datetime import datetime, timezone
import logging
import re
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stderr),
        logging.FileHandler("tool_calls.log")
    ]
)

logger = logging.getLogger(__name__)

def compute_authority_score(ds):
    downloads = ds.metadata.get("downloads", 0)
    likes = ds.metadata.get("likes", 0)

    # Log to avoid domination by very popular datasets
    return math.log(downloads + 1) + 0.2 * likes


def compute_recency_score(ds):
    updated = ds.metadata.get("updatedAt")
    if not updated:
        return 0.0

    days_old = (datetime.now(timezone.utc) - updated).days
    # 0–3 range, decays over ~3 years
    return max(0.0, 3.0 - days_old / 365.0)


def compute_hybrid_score(ds):
    semantic = ds.metadata.get("confidence", 0.0)   
    authority = compute_authority_score(ds)
    recency = compute_recency_score(ds)

    score = (
        0.6 * semantic +
        0.25 * authority +
        0.15 * recency
    )

    ds.metadata["rank_score"] = round(score, 4)
    return score

def pre_rank_datasets(datasets, top_k=5):
    for ds in datasets:
        compute_hybrid_score(ds)

    ranked = sorted(
        datasets,
        key=lambda d: d.metadata.get("rank_score", 0),
        reverse=True,
    )

    return ranked[:top_k]


def enrich_with_details(datasets, mcp_client, max_length=600):
    enriched = {}
    for ds in datasets:
        try:
            details = mcp_client.get_prompt(
                "Dataset Details", {"dataset_id": ds.id}
            )
            if details:
                # Truncate if too long
                if len(details) > max_length:
                    details = details[:max_length] + "..."
                    details = re.sub(r"See the full description.*$", "", details, flags=re.MULTILINE) 
                enriched[ds.id] = details
        except Exception as e:
            logger.error(f"Enrichment failed for {ds.id}: {e}")
    return enriched
