"""CLI entry point for forensic-astronomer."""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional

import click
import httpx
from rich.console import Console

from .models import AnalysisResult, Platform, SentimentAnalysisResult
from .parsers import parse_url
from .storage import (
    ensure_directories,
    get_output_directory,
    save_analysis_result,
    save_raw_responses,
    save_sentiment_results,
    load_analysis_result,
)
from .reports import generate_text_report, print_summary, save_report

console = Console()


def find_existing_data(url: str, directories: dict[str, Path]) -> AnalysisResult | None:
    """Check if we already have scraped data for this URL."""
    # Extract post ID from URL
    url_parts = url.rstrip("/").split("/")
    post_id = url_parts[-1] if url_parts else None
    if not post_id:
        return None

    # Check both platform directories
    for platform_name in ["bluesky", "twitter"]:
        platform_dir = directories.get(platform_name)
        if not platform_dir:
            continue

        response_file = platform_dir / f"{post_id}_responses.json"
        if response_file.exists():
            try:
                result = load_analysis_result(response_file)
                console.print(f"[cyan]Found existing data for {url}[/cyan]")
                return result
            except Exception as e:
                console.print(f"[yellow]Could not load existing data: {e}[/yellow]")

    return None


async def scrape_url(url: str, client: httpx.AsyncClient, include_likes: bool = False) -> Optional[AnalysisResult]:
    """Scrape a single URL and return the analysis result."""
    parsed = parse_url(url)
    if not parsed:
        console.print(f"[red]Could not parse URL: {url}[/red]")
        return None

    console.print(f"[blue]Scraping {parsed.platform.value}: {url}[/blue]")

    if parsed.platform == Platform.BLUESKY:
        from .bluesky_scraper import scrape_bluesky_backlinks

        return await scrape_bluesky_backlinks(parsed, client, include_likes=include_likes)
    elif parsed.platform == Platform.TWITTER:
        from .twitter_scraper import scrape_twitter_responses

        return await scrape_twitter_responses(parsed, client)

    return None


async def run_analysis(
    urls: list[str],
    output_dir: Path,
    generate_graphs: bool = False,
    run_sentiment: bool = False,
    force_rescrape: bool = False,
    include_likes: bool = False,
) -> tuple[list[AnalysisResult], dict[str, SentimentAnalysisResult]]:
    """Run the full analysis pipeline."""
    results: list[AnalysisResult] = []
    sentiment_results: dict[str, SentimentAnalysisResult] = {}
    sentiment_timeline_data: dict[str, list] = {}

    directories = ensure_directories(output_dir)

    async with httpx.AsyncClient(timeout=30.0) as client:
        for url in urls:
            try:
                # Check for existing data first (unless force rescrape)
                if not force_rescrape:
                    existing = find_existing_data(url, directories)
                    if existing:
                        results.append(existing)
                        continue

                result = await scrape_url(url, client, include_likes)
                if result:
                    results.append(result)

                    # Save results
                    filepath = save_analysis_result(result, directories)
                    console.print(f"[green]Saved to: {filepath}[/green]")

                    raw_filepath = save_raw_responses(result, directories)
                    console.print(f"[dim]Raw data saved to: {raw_filepath}[/dim]")
            except Exception as e:
                console.print(f"[red]Error scraping {url}: {e}[/red]")
                import traceback
                traceback.print_exc()

    # Run sentiment analysis if requested
    if run_sentiment and results:
        try:
            from .sentiment import analyze_sentiments, print_sentiment_summary

            for result in results:
                console.print(f"\n[blue]Running sentiment analysis for {result.platform.value}...[/blue]")
                sentiment, timeline, type_breakdowns = analyze_sentiments(result)
                sentiment_results[result.source_url] = sentiment
                sentiment_timeline_data[result.source_url] = timeline

                print_sentiment_summary(sentiment, type_breakdowns)

                # Re-save the main results file with sentiment data included
                filepath = save_analysis_result(result, directories)
                console.print(f"[green]Updated with sentiment: {filepath}[/green]")
        except ImportError as e:
            console.print(f"[yellow]Sentiment analysis unavailable: {e}[/yellow]")

    # Generate graphs if requested
    if generate_graphs and results:
        from .graphs import (
            generate_timeline_graph,
            generate_response_type_chart,
            generate_sentiment_timeline,
            generate_sentiment_comparison,
        )

        console.print("\n[blue]Generating graphs...[/blue]")

        # Timeline graph
        path = generate_timeline_graph(results, output_dir, align_start=True)
        console.print(f"[green]Timeline graph saved to: {path}[/green]")

        # Response type chart
        path = generate_response_type_chart(results, output_dir)
        console.print(f"[green]Response type chart saved to: {path}[/green]")

        # Sentiment graphs if sentiment was run
        if sentiment_results and sentiment_timeline_data:
            path = generate_sentiment_timeline(
                results, sentiment_timeline_data, output_dir, align_start=True
            )
            console.print(f"[green]Sentiment timeline saved to: {path}[/green]")

            path = generate_sentiment_comparison(sentiment_results, output_dir)
            if path:
                console.print(f"[green]Sentiment comparison saved to: {path}[/green]")

    # Generate and save report
    if results:
        print_summary(results)

        report_path = save_report(results, output_dir, sentiment_results)
        console.print(f"\n[green]Report saved to: {report_path}[/green]")

    return results, sentiment_results


@click.command()
@click.argument("urls", nargs=-1, required=True)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Output directory (default: ./forensic-astronomer-{datestamp})",
)
@click.option(
    "--graphs",
    "-g",
    is_flag=True,
    default=False,
    help="Generate timeline graphs (requires matplotlib)",
)
@click.option(
    "--sentiment",
    "-s",
    is_flag=True,
    default=False,
    help="Run sentiment analysis (requires transformers)",
)
@click.option(
    "--twitter-token",
    envvar="TWITTER_BEARER_TOKEN",
    help="Twitter API bearer token (or set TWITTER_BEARER_TOKEN env var)",
)
@click.option(
    "--force",
    "-f",
    is_flag=True,
    default=False,
    help="Force re-scraping even if data exists in output directory",
)
@click.option(
    "--include-likes",
    is_flag=True,
    default=False,
    help="Include likes in the analysis (skipped by default as they have no text)",
)
def main(
    urls: tuple[str, ...],
    output: Optional[Path],
    graphs: bool,
    sentiment: bool,
    twitter_token: Optional[str],
    force: bool,
    include_likes: bool,
):
    """Analyze cross-platform interactions for Twitter and Bluesky posts.

    Provide one or more URLs to Twitter/X or Bluesky posts to analyze.

    Examples:

        # Analyze a Bluesky post
        forensic-astronomer https://bsky.app/profile/user.bsky.social/post/abc123

        # Analyze a Twitter post
        forensic-astronomer https://twitter.com/user/status/1234567890

        # Analyze both with graphs and sentiment
        forensic-astronomer -g -s URL1 URL2
    """
    if twitter_token:
        import os
        os.environ["TWITTER_BEARER_TOKEN"] = twitter_token

    # Check for Twitter URLs without token
    has_twitter = any("twitter.com" in url or "x.com" in url for url in urls)
    if has_twitter and not twitter_token:
        import os
        if not os.environ.get("TWITTER_BEARER_TOKEN"):
            console.print(
                "[yellow]Warning: Twitter URL detected but no TWITTER_BEARER_TOKEN set. "
                "Twitter scraping will fail.[/yellow]"
            )

    output_dir = output or get_output_directory()

    console.print(f"[bold]Forensic Astronomer[/bold]")
    console.print(f"Output directory: {output_dir}")
    console.print(f"URLs to analyze: {len(urls)}")
    console.print(f"Graphs: {'Yes' if graphs else 'No'}")
    console.print(f"Sentiment: {'Yes' if sentiment else 'No'}")
    console.print(f"Include likes: {'Yes' if include_likes else 'No'}")
    console.print(f"Use cached data: {'No (--force)' if force else 'Yes'}")
    console.print()

    try:
        asyncio.run(run_analysis(list(urls), output_dir, graphs, sentiment, force, include_likes))
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        import traceback
        traceback.print_exc()
        raise SystemExit(1)


if __name__ == "__main__":
    main()
