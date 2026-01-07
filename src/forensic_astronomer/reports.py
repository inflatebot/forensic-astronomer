"""Report generation for cross-platform interaction analysis."""

from datetime import datetime
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.table import Table

from .models import AnalysisResult, Platform, SentimentAnalysisResult

console = Console()


def generate_summary_table(results: list[AnalysisResult]) -> Table:
    """Generate a summary table of all analysis results."""
    table = Table(title="Cross-Platform Interaction Summary")

    table.add_column("Platform", style="cyan")
    table.add_column("Source URL", style="blue")
    table.add_column("Total Responses", justify="right", style="green")
    table.add_column("Replies", justify="right")
    table.add_column("Quotes", justify="right")
    table.add_column("Reposts", justify="right")
    table.add_column("Mentions", justify="right")

    for result in results:
        table.add_row(
            result.platform.value.title(),
            result.source_url[:50] + "..." if len(result.source_url) > 50 else result.source_url,
            str(result.total_responses),
            str(result.responses_by_type.get("reply", 0)),
            str(result.responses_by_type.get("quote", 0)),
            str(result.responses_by_type.get("repost", 0)),
            str(result.responses_by_type.get("mention", 0)),
        )

    return table


def generate_text_report(
    results: list[AnalysisResult],
    sentiment_results: Optional[dict[str, SentimentAnalysisResult]] = None,
) -> str:
    """Generate a text-based report of the analysis."""
    lines = []
    lines.append("=" * 80)
    lines.append("CROSS-PLATFORM INTERACTION ANALYSIS REPORT")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 80)
    lines.append("")

    # Summary statistics
    total_responses = sum(r.total_responses for r in results)
    total_replies = sum(r.responses_by_type.get("reply", 0) for r in results)
    total_quotes = sum(r.responses_by_type.get("quote", 0) for r in results)
    total_reposts = sum(r.responses_by_type.get("repost", 0) for r in results)
    total_mentions = sum(r.responses_by_type.get("mention", 0) for r in results)

    lines.append("OVERALL SUMMARY")
    lines.append("-" * 40)
    lines.append(f"Total responses across all platforms: {total_responses}")
    lines.append(f"  - Replies:  {total_replies}")
    lines.append(f"  - Quotes:   {total_quotes}")
    lines.append(f"  - Reposts:  {total_reposts}")
    lines.append(f"  - Mentions: {total_mentions}")
    lines.append("")

    # Per-platform breakdown
    for result in results:
        lines.append(f"PLATFORM: {result.platform.value.upper()}")
        lines.append("-" * 40)
        lines.append(f"Source URL: {result.source_url}")
        lines.append(f"Fetched at: {result.fetched_at.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Total responses: {result.total_responses}")
        lines.append("")
        lines.append("Response breakdown:")

        for response_type, count in sorted(result.responses_by_type.items()):
            percentage = (count / result.total_responses * 100) if result.total_responses > 0 else 0
            lines.append(f"  - {response_type.title()}: {count} ({percentage:.1f}%)")

        lines.append("")

        # Timeline summary
        if result.responses:
            sorted_responses = sorted(result.responses, key=lambda r: r.created_at)
            first_response = sorted_responses[0]
            last_response = sorted_responses[-1]

            lines.append("Timeline:")
            lines.append(f"  First response: {first_response.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
            lines.append(f"  Last response:  {last_response.created_at.strftime('%Y-%m-%d %H:%M:%S')}")

            duration = last_response.created_at - first_response.created_at
            hours = duration.total_seconds() / 3600
            lines.append(f"  Duration: {hours:.1f} hours")
            lines.append("")

        # Top responders
        responder_counts: dict[str, int] = {}
        for response in result.responses:
            handle = response.author_handle
            responder_counts[handle] = responder_counts.get(handle, 0) + 1

        if responder_counts:
            top_responders = sorted(responder_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            lines.append("Top responders:")
            for handle, count in top_responders:
                lines.append(f"  @{handle}: {count} response(s)")
            lines.append("")

        # Sentiment summary if available
        if sentiment_results and result.source_url in sentiment_results:
            sentiment = sentiment_results[result.source_url]
            lines.append("Sentiment Analysis:")
            lines.append(f"  Analyzed: {sentiment.total_analyzed} responses")
            for label, count in sorted(sentiment.sentiment_counts.items()):
                percentage = (count / sentiment.total_analyzed * 100) if sentiment.total_analyzed > 0 else 0
                lines.append(f"  - {label.title()}: {count} ({percentage:.1f}%)")
            lines.append("")

    # Cross-platform comparison (if both platforms present)
    twitter_results = [r for r in results if r.platform == Platform.TWITTER]
    bluesky_results = [r for r in results if r.platform == Platform.BLUESKY]

    if twitter_results and bluesky_results:
        lines.append("CROSS-PLATFORM COMPARISON")
        lines.append("-" * 40)

        twitter_total = sum(r.total_responses for r in twitter_results)
        bluesky_total = sum(r.total_responses for r in bluesky_results)

        lines.append(f"Twitter total responses:  {twitter_total}")
        lines.append(f"Bluesky total responses:  {bluesky_total}")

        if twitter_total > 0 and bluesky_total > 0:
            ratio = twitter_total / bluesky_total
            lines.append(f"Twitter:Bluesky ratio:    {ratio:.2f}:1")
        lines.append("")

    lines.append("=" * 80)
    lines.append("END OF REPORT")
    lines.append("=" * 80)

    return "\n".join(lines)


def save_report(
    results: list[AnalysisResult],
    output_dir: Path,
    sentiment_results: Optional[dict[str, SentimentAnalysisResult]] = None,
) -> Path:
    """Save the report to the output directory."""
    report_text = generate_text_report(results, sentiment_results)

    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = reports_dir / f"analysis_report_{timestamp}.txt"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    return report_path


def print_summary(results: list[AnalysisResult]):
    """Print a summary to the console."""
    table = generate_summary_table(results)
    console.print(table)
    console.print()

    # Print totals
    total = sum(r.total_responses for r in results)
    console.print(f"[bold green]Total responses across all sources: {total}[/bold green]")
