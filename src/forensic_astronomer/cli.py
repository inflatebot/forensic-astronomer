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
)
from .reports import generate_text_report, print_summary, save_report

console = Console()


async def scrape_url(url: str, client: httpx.AsyncClient) -> Optional[AnalysisResult]:
    """Scrape a single URL and return the analysis result."""
    parsed = parse_url(url)
    if not parsed:
        console.print(f"[red]Could not parse URL: {url}[/red]")
        return None

    console.print(f"[blue]Scraping {parsed.platform.value}: {url}[/blue]")

    if parsed.platform == Platform.BLUESKY:
        from .bluesky_scraper import scrape_bluesky_backlinks

        return await scrape_bluesky_backlinks(parsed, client)
    elif parsed.platform == Platform.TWITTER:
        from .twitter_scraper import scrape_twitter_responses

        return await scrape_twitter_responses(parsed, client)

    return None


async def run_analysis(
    urls: list[str],
    output_dir: Path,
    generate_graphs: bool = False,
    run_sentiment: bool = False,
) -> tuple[list[AnalysisResult], dict[str, SentimentAnalysisResult]]:
    """Run the full analysis pipeline."""
    results: list[AnalysisResult] = []
    sentiment_results: dict[str, SentimentAnalysisResult] = {}
    sentiment_timeline_data: dict[str, list] = {}

    directories = ensure_directories(output_dir)

    async with httpx.AsyncClient(timeout=30.0) as client:
        for url in urls:
            try:
                result = await scrape_url(url, client)
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
                sentiment, timeline = analyze_sentiments(result)
                sentiment_results[result.source_url] = sentiment
                sentiment_timeline_data[result.source_url] = timeline

                print_sentiment_summary(sentiment)

                # Extract post ID for filename
                url_parts = result.source_url.rstrip("/").split("/")
                post_id = url_parts[-1] if url_parts else "unknown"

                filepath = save_sentiment_results(
                    sentiment, result.platform, post_id, directories
                )
                console.print(f"[green]Sentiment saved to: {filepath}[/green]")
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
def main(
    urls: tuple[str, ...],
    output: Optional[Path],
    graphs: bool,
    sentiment: bool,
    twitter_token: Optional[str],
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
    console.print()

    try:
        asyncio.run(run_analysis(list(urls), output_dir, graphs, sentiment))
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        import traceback
        traceback.print_exc()
        raise SystemExit(1)


if __name__ == "__main__":
    main()
