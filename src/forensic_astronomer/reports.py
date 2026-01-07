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
    table.add_column("Thread", justify="right")
    table.add_column("Quotes", justify="right")
    table.add_column("Reposts", justify="right")
    table.add_column("Mentions", justify="right")

    for result in results:
        table.add_row(
            result.platform.value.title(),
            result.source_url[:50] + "..." if len(result.source_url) > 50 else result.source_url,
            str(result.total_responses),
            str(result.responses_by_type.get("reply", 0)),
            str(result.responses_by_type.get("thread_reply", 0)),
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
    total_thread_replies = sum(r.responses_by_type.get("thread_reply", 0) for r in results)
    total_quotes = sum(r.responses_by_type.get("quote", 0) for r in results)
    total_reposts = sum(r.responses_by_type.get("repost", 0) for r in results)
    total_mentions = sum(r.responses_by_type.get("mention", 0) for r in results)
    total_likes = sum(r.responses_by_type.get("like", 0) for r in results)

    lines.append("OVERALL SUMMARY")
    lines.append("-" * 40)
    lines.append(f"Total responses across all platforms: {total_responses}")
    lines.append(f"  - Replies (to OP):    {total_replies}")
    lines.append(f"  - Thread replies:     {total_thread_replies}")
    lines.append(f"  - Quotes:             {total_quotes}")
    lines.append(f"  - Reposts:            {total_reposts}")
    lines.append(f"  - Mentions:           {total_mentions}")
    if total_likes > 0:
        lines.append(f"  - Likes:              {total_likes}")
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
            lines.append("")
            lines.append("  Overall:")
            for label in ["positive", "neutral", "negative"]:
                count = sentiment.sentiment_counts.get(label, 0)
                percentage = (count / sentiment.total_analyzed * 100) if sentiment.total_analyzed > 0 else 0
                lines.append(f"    - {label.title()}: {count} ({percentage:.1f}%)")
            lines.append("")

            # Sentiment by response type (computed from Response objects)
            type_display_names = {
                "reply": "Direct Replies (to OP)",
                "thread_reply": "Thread Replies (to others)",
                "quote": "Quote Posts",
                "repost": "Reposts",
                "mention": "Mentions",
                "like": "Likes",
            }

            # Group responses by type and compute sentiment
            responses_with_sentiment = [r for r in result.responses if r.sentiment_label]
            if responses_with_sentiment:
                lines.append("  By Response Type:")

                # Collect sentiments by response type
                type_sentiments: dict[str, list[str]] = {}
                for response in responses_with_sentiment:
                    type_name = response.response_type.value
                    if type_name not in type_sentiments:
                        type_sentiments[type_name] = []
                    type_sentiments[type_name].append(response.sentiment_label)

                for type_name in ["reply", "thread_reply", "quote", "repost", "mention", "like"]:
                    if type_name not in type_sentiments:
                        continue

                    sentiments = type_sentiments[type_name]
                    display_name = type_display_names.get(type_name, type_name.title())
                    total = len(sentiments)

                    lines.append(f"    {display_name} ({total}):")

                    for label in ["positive", "neutral", "negative"]:
                        count = sum(1 for s in sentiments if s == label)
                        percentage = (count / total * 100) if total > 0 else 0
                        lines.append(f"      - {label.title()}: {count} ({percentage:.1f}%)")

            lines.append("")

        # LLM Analysis summary (if available)
        responses_with_llm = [r for r in result.responses if r.llm_intent is not None]
        if responses_with_llm:
            lines.append("LLM Analysis:")
            lines.append(f"  Analyzed: {len(responses_with_llm)} responses")
            lines.append("")

            # Upset breakdown
            upset_responses = [r for r in responses_with_llm if r.llm_is_upset]
            upset_pct = len(upset_responses) / len(responses_with_llm) * 100
            lines.append(f"  Upset responses: {len(upset_responses)} ({upset_pct:.1f}%)")

            if upset_responses:
                # Upset at breakdown
                upset_at_counts: dict[str, int] = {}
                for r in upset_responses:
                    target = r.llm_upset_at or "unclear"
                    upset_at_counts[target] = upset_at_counts.get(target, 0) + 1

                upset_at_names = {
                    "op": "At the OP (critical)",
                    "on_behalf_of_op": "On OP's behalf (supportive)",
                    "third_party": "At third party",
                    "situation": "At the situation",
                    "unclear": "Unclear",
                }
                lines.append("  Upset at:")
                for target, count in sorted(upset_at_counts.items(), key=lambda x: -x[1]):
                    pct = count / len(upset_responses) * 100
                    name = upset_at_names.get(target, target)
                    lines.append(f"    - {name}: {count} ({pct:.1f}%)")
            lines.append("")

            # Intent breakdown
            intent_counts: dict[str, int] = {}
            for r in responses_with_llm:
                intent = r.llm_intent or "other"
                intent_counts[intent] = intent_counts.get(intent, 0) + 1

            intent_names = {
                "emotional_expression": "Emotional expression",
                "rhetorical_point": "Making a point",
                "question": "Asking question",
                "agreement": "Agreement",
                "disagreement": "Disagreement",
                "humor": "Humor",
                "information": "Sharing info",
                "other": "Other",
            }
            lines.append("  Intent:")
            for intent, count in sorted(intent_counts.items(), key=lambda x: -x[1]):
                pct = count / len(responses_with_llm) * 100
                name = intent_names.get(intent, intent)
                lines.append(f"    - {name}: {count} ({pct:.1f}%)")
            lines.append("")

            # Stance breakdown
            stance_counts: dict[str, int] = {}
            for r in responses_with_llm:
                stance = r.llm_stance or "neutral"
                stance_counts[stance] = stance_counts.get(stance, 0) + 1

            lines.append("  Stance toward OP:")
            for stance, count in sorted(stance_counts.items(), key=lambda x: -x[1]):
                pct = count / len(responses_with_llm) * 100
                lines.append(f"    - {stance.title()}: {count} ({pct:.1f}%)")
            lines.append("")

            # Rhetorical points
            rhetorical = [r for r in responses_with_llm if r.llm_rhetorical_point]
            if rhetorical:
                lines.append(f"  Rhetorical points made ({len(rhetorical)}):")
                for r in rhetorical[:10]:
                    lines.append(f"    - {r.llm_rhetorical_point}")
                if len(rhetorical) > 10:
                    lines.append(f"    ... and {len(rhetorical) - 10} more")
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
