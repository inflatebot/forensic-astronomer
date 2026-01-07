"""Sentiment analysis for cross-platform interaction analysis."""

from datetime import datetime
from typing import Optional

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from transformers import pipeline

from .models import AnalysisResult, Response, SentimentAnalysisResult, SentimentScore

console = Console()


def get_sentiment_pipeline():
    """Get or create the sentiment analysis pipeline."""
    # Use a small, efficient sentiment model
    # distilbert-base-uncased-finetuned-sst-2-english is fast and accurate
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=-1,  # CPU, use 0 for GPU
    )


def map_sentiment_label(label: str) -> str:
    """Map model output labels to standard labels."""
    label = label.upper()
    if label in ("POSITIVE", "POS", "1", "LABEL_1"):
        return "positive"
    elif label in ("NEGATIVE", "NEG", "0", "LABEL_0"):
        return "negative"
    else:
        return "neutral"


def analyze_response_sentiment(
    response: Response,
    pipeline,
) -> Optional[SentimentScore]:
    """Analyze sentiment of a single response."""
    if not response.text:
        return None

    try:
        # Truncate text to 512 tokens (model limit)
        text = response.text[:512]

        result = pipeline(text)[0]

        return SentimentScore(
            response_id=response.id,
            label=map_sentiment_label(result["label"]),
            score=result["score"],
        )
    except Exception as e:
        console.print(f"[yellow]Warning: Could not analyze sentiment for {response.id}: {e}[/yellow]")
        return None


def analyze_sentiments(
    result: AnalysisResult,
    progress_callback=None,
) -> tuple[SentimentAnalysisResult, list[tuple[datetime, SentimentScore]], dict[str, dict]]:
    """Analyze sentiment for all responses in an analysis result.

    Returns:
        Tuple of:
        - SentimentAnalysisResult: Overall sentiment results
        - list of (datetime, sentiment) pairs for timeline
        - dict mapping response_type to sentiment breakdown
    """
    console.print("[blue]Loading sentiment analysis model...[/blue]")
    sentiment_pipeline = get_sentiment_pipeline()
    console.print("[green]Model loaded.[/green]")

    sentiments: list[SentimentScore] = []
    timeline_data: list[tuple[datetime, SentimentScore]] = []

    # Track sentiments by response type
    sentiments_by_type: dict[str, list[tuple[Response, SentimentScore]]] = {}

    responses_with_text = [r for r in result.responses if r.text]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        task = progress.add_task(
            f"Analyzing {len(responses_with_text)} responses...",
            total=len(responses_with_text),
        )

        for response in responses_with_text:
            sentiment = analyze_response_sentiment(response, sentiment_pipeline)
            if sentiment:
                sentiments.append(sentiment)
                timeline_data.append((response.created_at, sentiment))

                # Group by response type
                type_name = response.response_type.value
                if type_name not in sentiments_by_type:
                    sentiments_by_type[type_name] = []
                sentiments_by_type[type_name].append((response, sentiment))

            progress.advance(task)

    # Count overall sentiments
    sentiment_counts: dict[str, int] = {}
    score_sums: dict[str, float] = {}

    for sentiment in sentiments:
        label = sentiment.label
        sentiment_counts[label] = sentiment_counts.get(label, 0) + 1
        score_sums[label] = score_sums.get(label, 0) + sentiment.score

    # Calculate average scores
    average_scores = {
        label: score_sums[label] / sentiment_counts[label]
        for label in sentiment_counts
    }

    # Calculate sentiment breakdown by response type
    type_breakdowns: dict[str, dict] = {}
    for type_name, type_sentiments in sentiments_by_type.items():
        type_counts: dict[str, int] = {}
        type_score_sums: dict[str, float] = {}

        for response, sentiment in type_sentiments:
            label = sentiment.label
            type_counts[label] = type_counts.get(label, 0) + 1
            type_score_sums[label] = type_score_sums.get(label, 0) + sentiment.score

        type_avg_scores = {
            label: type_score_sums[label] / type_counts[label]
            for label in type_counts
        }

        type_breakdowns[type_name] = {
            "total": len(type_sentiments),
            "counts": type_counts,
            "average_scores": type_avg_scores,
        }

    analysis_result = SentimentAnalysisResult(
        total_analyzed=len(sentiments),
        sentiment_counts=sentiment_counts,
        average_scores=average_scores,
        sentiments=sentiments,
    )

    return analysis_result, timeline_data, type_breakdowns


def print_sentiment_summary(
    result: SentimentAnalysisResult,
    type_breakdowns: dict[str, dict] | None = None,
):
    """Print a summary of sentiment analysis to the console."""
    console.print()
    console.print("[bold]Sentiment Analysis Summary[/bold]")
    console.print(f"Total analyzed: {result.total_analyzed}")
    console.print()

    # Overall sentiment
    console.print("[bold]Overall:[/bold]")
    for label in ["positive", "neutral", "negative"]:
        count = result.sentiment_counts.get(label, 0)
        percentage = (count / result.total_analyzed * 100) if result.total_analyzed > 0 else 0
        avg_score = result.average_scores.get(label, 0)

        color = {
            "positive": "green",
            "neutral": "white",
            "negative": "red",
        }.get(label, "white")

        console.print(
            f"  [{color}]{label.title()}[/{color}]: {count} ({percentage:.1f}%) - "
            f"avg confidence: {avg_score:.2f}"
        )

    # Breakdown by response type
    if type_breakdowns:
        console.print()
        console.print("[bold]By Response Type:[/bold]")

        type_display_names = {
            "reply": "Replies",
            "quote": "Quote Posts",
            "repost": "Reposts",
            "mention": "Mentions",
            "like": "Likes",
        }

        for type_name in ["reply", "quote", "repost", "mention", "like"]:
            if type_name not in type_breakdowns:
                continue

            breakdown = type_breakdowns[type_name]
            display_name = type_display_names.get(type_name, type_name.title())
            total = breakdown["total"]

            console.print(f"\n  [cyan]{display_name}[/cyan] ({total} analyzed):")

            for label in ["positive", "neutral", "negative"]:
                count = breakdown["counts"].get(label, 0)
                percentage = (count / total * 100) if total > 0 else 0

                color = {
                    "positive": "green",
                    "neutral": "white",
                    "negative": "red",
                }.get(label, "white")

                console.print(
                    f"    [{color}]{label.title()}[/{color}]: {count} ({percentage:.1f}%)"
                )
