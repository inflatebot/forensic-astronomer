"""Graph generation for cross-platform interaction analysis."""

from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

from .models import AnalysisResult, Platform, SentimentAnalysisResult, SentimentScore


def generate_timeline_graph(
    results: list[AnalysisResult],
    output_dir: Path,
    align_start: bool = True,
) -> Path:
    """Generate a timeline graph of responses over time.

    Args:
        results: List of analysis results
        output_dir: Directory to save the graph
        align_start: If True, align the start times of different platforms

    Returns:
        Path to the saved graph
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=align_start)

    colors = {
        Platform.TWITTER: "#1DA1F2",  # Twitter blue
        Platform.BLUESKY: "#0085FF",  # Bluesky blue
    }

    platform_data: dict[Platform, list[datetime]] = defaultdict(list)
    platform_starts: dict[Platform, datetime] = {}

    # Collect all response times by platform
    for result in results:
        times = [r.created_at for r in result.responses]
        if times:
            platform_data[result.platform].extend(times)
            if result.platform not in platform_starts:
                platform_starts[result.platform] = min(times)
            else:
                platform_starts[result.platform] = min(
                    platform_starts[result.platform], min(times)
                )

    # Calculate time bins (hourly)
    for idx, (platform, times) in enumerate(platform_data.items()):
        ax = axes[idx] if len(platform_data) > 1 else axes

        if not times:
            continue

        times = sorted(times)

        if align_start and platform_starts:
            # Shift times so they start at the same point
            earliest_start = min(platform_starts.values())
            platform_start = platform_starts[platform]
            offset = platform_start - earliest_start
            times = [t - offset for t in times]

        # Create hourly bins
        start_time = min(times)
        end_time = max(times)
        duration = end_time - start_time

        # Determine bin size based on duration
        if duration.total_seconds() < 6 * 3600:  # Less than 6 hours
            bin_size = timedelta(minutes=15)
            date_format = "%H:%M"
        elif duration.total_seconds() < 24 * 3600:  # Less than 24 hours
            bin_size = timedelta(hours=1)
            date_format = "%H:%M"
        elif duration.total_seconds() < 7 * 24 * 3600:  # Less than a week
            bin_size = timedelta(hours=4)
            date_format = "%m/%d %H:%M"
        else:
            bin_size = timedelta(days=1)
            date_format = "%m/%d"

        # Count responses per bin
        bins: dict[datetime, int] = defaultdict(int)
        for t in times:
            # Round down to bin
            bin_start = start_time + timedelta(
                seconds=(
                    (t - start_time).total_seconds() // bin_size.total_seconds()
                ) * bin_size.total_seconds()
            )
            bins[bin_start] += 1

        # Create cumulative counts
        sorted_bins = sorted(bins.items())
        bin_times = [b[0] for b in sorted_bins]
        bin_counts = [b[1] for b in sorted_bins]

        # Calculate cumulative
        cumulative = []
        total = 0
        for count in bin_counts:
            total += count
            cumulative.append(total)

        # Plot
        ax.bar(bin_times, bin_counts, width=bin_size.total_seconds() / 86400,
               color=colors.get(platform, "gray"), alpha=0.7, label=f"{platform.value.title()} (per bin)")
        ax.plot(bin_times, cumulative, color="red", linewidth=2, label="Cumulative")

        ax.set_title(f"{platform.value.title()} Responses Over Time")
        ax.set_xlabel("Time" + (" (aligned)" if align_start else ""))
        ax.set_ylabel("Number of Responses")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter(date_format))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.tight_layout()

    graphs_dir = output_dir / "graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    graph_path = graphs_dir / f"timeline_{timestamp}.png"
    plt.savefig(graph_path, dpi=150, bbox_inches="tight")
    plt.close()

    return graph_path


def generate_response_type_chart(
    results: list[AnalysisResult],
    output_dir: Path,
) -> Path:
    """Generate a pie chart of response types."""
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 6))

    if len(results) == 1:
        axes = [axes]

    colors = {
        "reply": "#4CAF50",
        "quote": "#2196F3",
        "repost": "#FF9800",
        "mention": "#9C27B0",
        "like": "#E91E63",
    }

    for ax, result in zip(axes, results):
        if not result.responses_by_type:
            continue

        labels = []
        sizes = []
        chart_colors = []

        for response_type, count in result.responses_by_type.items():
            labels.append(f"{response_type.title()}\n({count})")
            sizes.append(count)
            chart_colors.append(colors.get(response_type, "#999999"))

        ax.pie(sizes, labels=labels, colors=chart_colors, autopct="%1.1f%%",
               startangle=90)
        ax.set_title(f"{result.platform.value.title()} Response Types")

    plt.tight_layout()

    graphs_dir = output_dir / "graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    graph_path = graphs_dir / f"response_types_{timestamp}.png"
    plt.savefig(graph_path, dpi=150, bbox_inches="tight")
    plt.close()

    return graph_path


def generate_sentiment_timeline(
    results: list[AnalysisResult],
    sentiment_data: dict[str, list[tuple[datetime, SentimentScore]]],
    output_dir: Path,
    align_start: bool = True,
) -> Path:
    """Generate a sentiment timeline graph.

    Args:
        results: List of analysis results
        sentiment_data: Dict mapping source_url to list of (datetime, sentiment) tuples
        output_dir: Directory to save the graph
        align_start: If True, align the start times of different platforms

    Returns:
        Path to the saved graph
    """
    fig, axes = plt.subplots(len(results), 1, figsize=(14, 5 * len(results)))

    if len(results) == 1:
        axes = [axes]

    sentiment_colors = {
        "positive": "#4CAF50",
        "negative": "#F44336",
        "neutral": "#9E9E9E",
    }

    # Find earliest start for alignment
    all_starts = []
    for result in results:
        if result.source_url in sentiment_data:
            times = [t for t, _ in sentiment_data[result.source_url]]
            if times:
                all_starts.append(min(times))

    earliest_start = min(all_starts) if all_starts else datetime.now()

    for ax, result in zip(axes, results):
        if result.source_url not in sentiment_data:
            continue

        data = sentiment_data[result.source_url]
        if not data:
            continue

        # Sort by time
        data = sorted(data, key=lambda x: x[0])

        times = [t for t, _ in data]
        sentiments = [s for _, s in data]

        if align_start:
            # Shift times
            start = min(times)
            offset = start - earliest_start
            times = [t - offset for t in times]

        # Calculate rolling sentiment (window of 10 responses)
        window_size = min(10, len(sentiments))

        # Map sentiment to numeric: positive=1, neutral=0, negative=-1
        sentiment_map = {"positive": 1, "neutral": 0, "negative": -1}
        numeric_sentiments = [sentiment_map.get(s.label, 0) for s in sentiments]

        # Calculate rolling average
        rolling_avg = []
        for i in range(len(numeric_sentiments)):
            start_idx = max(0, i - window_size + 1)
            window = numeric_sentiments[start_idx:i + 1]
            rolling_avg.append(sum(window) / len(window))

        # Plot scatter of individual sentiments
        for label, color in sentiment_colors.items():
            label_times = [t for t, s in zip(times, sentiments) if s.label == label]
            label_values = [sentiment_map[label]] * len(label_times)
            ax.scatter(label_times, label_values, c=color, alpha=0.3, s=20, label=label.title())

        # Plot rolling average
        ax.plot(times, rolling_avg, color="blue", linewidth=2, label=f"Rolling Avg (n={window_size})")

        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.set_title(f"{result.platform.value.title()} Sentiment Over Time")
        ax.set_xlabel("Time" + (" (aligned)" if align_start else ""))
        ax.set_ylabel("Sentiment (-1 to 1)")
        ax.set_ylim(-1.5, 1.5)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.tight_layout()

    graphs_dir = output_dir / "graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    graph_path = graphs_dir / f"sentiment_timeline_{timestamp}.png"
    plt.savefig(graph_path, dpi=150, bbox_inches="tight")
    plt.close()

    return graph_path


def generate_sentiment_comparison(
    sentiment_results: dict[str, SentimentAnalysisResult],
    output_dir: Path,
) -> Path | None:
    """Generate a comparison chart of sentiment across platforms."""
    if not sentiment_results:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))

    platforms = list(sentiment_results.keys())
    sentiments = ["positive", "neutral", "negative"]
    x = np.arange(len(platforms))
    width = 0.25

    colors = {
        "positive": "#4CAF50",
        "neutral": "#9E9E9E",
        "negative": "#F44336",
    }

    for i, sentiment in enumerate(sentiments):
        values = []
        for platform in platforms:
            result = sentiment_results[platform]
            total = result.total_analyzed
            count = result.sentiment_counts.get(sentiment, 0)
            percentage = (count / total * 100) if total > 0 else 0
            values.append(percentage)

        ax.bar(x + i * width, values, width, label=sentiment.title(),
               color=colors[sentiment])

    ax.set_xlabel("Source")
    ax.set_ylabel("Percentage")
    ax.set_title("Sentiment Distribution Comparison")
    ax.set_xticks(x + width)
    ax.set_xticklabels([p[:30] + "..." if len(p) > 30 else p for p in platforms], rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    graphs_dir = output_dir / "graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    graph_path = graphs_dir / f"sentiment_comparison_{timestamp}.png"
    plt.savefig(graph_path, dpi=150, bbox_inches="tight")
    plt.close()

    return graph_path
