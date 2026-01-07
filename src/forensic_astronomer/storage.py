"""Data storage module for cross-platform interaction analysis."""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from .models import AnalysisResult, Platform, SentimentAnalysisResult


def get_output_directory(base_path: Optional[Path] = None) -> Path:
    """Get the output directory with datestamp."""
    if base_path is None:
        base_path = Path.cwd()

    datestamp = datetime.now().strftime("%Y%m%d")
    output_dir = base_path / f"forensic-astronomer-{datestamp}"
    return output_dir


def ensure_directories(output_dir: Path) -> dict[str, Path]:
    """Create the directory structure and return paths."""
    twitter_dir = output_dir / "twitter"
    bluesky_dir = output_dir / "bluesky"
    reports_dir = output_dir / "reports"
    graphs_dir = output_dir / "graphs"

    for directory in [twitter_dir, bluesky_dir, reports_dir, graphs_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    return {
        "root": output_dir,
        "twitter": twitter_dir,
        "bluesky": bluesky_dir,
        "reports": reports_dir,
        "graphs": graphs_dir,
    }


def save_analysis_result(result: AnalysisResult, directories: dict[str, Path]) -> Path:
    """Save an analysis result to the appropriate platform directory."""
    if result.platform == Platform.TWITTER:
        platform_dir = directories["twitter"]
    else:
        platform_dir = directories["bluesky"]

    # Generate a filename based on the source URL
    # Extract post ID from the URL for a clean filename
    url_parts = result.source_url.rstrip("/").split("/")
    post_id = url_parts[-1] if url_parts else "unknown"

    filename = f"{post_id}_responses.json"
    filepath = platform_dir / filename

    # Convert to serializable format
    data = {
        "source_url": result.source_url,
        "platform": result.platform.value,
        "total_responses": result.total_responses,
        "responses_by_type": result.responses_by_type,
        "fetched_at": result.fetched_at.isoformat(),
        "responses": [
            {
                "id": r.id,
                "platform": r.platform.value,
                "response_type": r.response_type.value,
                "author_handle": r.author_handle,
                "author_id": r.author_id,
                "text": r.text,
                "created_at": r.created_at.isoformat(),
                "parent_id": r.parent_id,
                "url": r.url,
            }
            for r in result.responses
        ],
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return filepath


def save_raw_responses(result: AnalysisResult, directories: dict[str, Path]) -> Path:
    """Save raw response data (including full API responses) for detailed analysis."""
    if result.platform == Platform.TWITTER:
        platform_dir = directories["twitter"]
    else:
        platform_dir = directories["bluesky"]

    url_parts = result.source_url.rstrip("/").split("/")
    post_id = url_parts[-1] if url_parts else "unknown"

    filename = f"{post_id}_raw.json"
    filepath = platform_dir / filename

    raw_data = [r.raw_data for r in result.responses if r.raw_data]

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(raw_data, f, indent=2, ensure_ascii=False)

    return filepath


def save_sentiment_results(
    sentiment: SentimentAnalysisResult,
    platform: Platform,
    post_id: str,
    directories: dict[str, Path],
) -> Path:
    """Save sentiment analysis results."""
    if platform == Platform.TWITTER:
        platform_dir = directories["twitter"]
    else:
        platform_dir = directories["bluesky"]

    filename = f"{post_id}_sentiment.json"
    filepath = platform_dir / filename

    data = {
        "total_analyzed": sentiment.total_analyzed,
        "sentiment_counts": sentiment.sentiment_counts,
        "average_scores": sentiment.average_scores,
        "sentiments": [
            {
                "response_id": s.response_id,
                "label": s.label,
                "score": s.score,
            }
            for s in sentiment.sentiments
        ],
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return filepath


def load_analysis_result(filepath: Path) -> AnalysisResult:
    """Load an analysis result from a JSON file."""
    from .models import Response, ResponseType

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    responses = [
        Response(
            id=r["id"],
            platform=Platform(r["platform"]),
            response_type=ResponseType(r["response_type"]),
            author_handle=r["author_handle"],
            author_id=r["author_id"],
            text=r.get("text"),
            created_at=datetime.fromisoformat(r["created_at"]),
            parent_id=r["parent_id"],
            url=r.get("url"),
        )
        for r in data["responses"]
    ]

    return AnalysisResult(
        source_url=data["source_url"],
        platform=Platform(data["platform"]),
        total_responses=data["total_responses"],
        responses_by_type=data["responses_by_type"],
        responses=responses,
        fetched_at=datetime.fromisoformat(data["fetched_at"]),
    )
