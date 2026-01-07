"""Twitter replies and quotes scraper with rate limiting."""

import asyncio
import os
import time
from datetime import datetime
from typing import Optional

import httpx
from rich.console import Console

from .models import AnalysisResult, ParsedURL, Platform, Response, ResponseType

console = Console()

# Twitter API v2 endpoints
TWITTER_API_BASE = "https://api.twitter.com/2"

# Rate limit configuration
# Twitter API v2 Basic tier: 10,000 tweets/month, 100 requests/15 min for search
RATE_LIMIT_REQUESTS = 100
RATE_LIMIT_WINDOW = 15 * 60  # 15 minutes in seconds
REQUEST_DELAY = RATE_LIMIT_WINDOW / RATE_LIMIT_REQUESTS  # ~9 seconds between requests


class RateLimiter:
    """Simple rate limiter for Twitter API."""

    def __init__(self, requests_per_window: int, window_seconds: int):
        self.requests_per_window = requests_per_window
        self.window_seconds = window_seconds
        self.request_times: list[float] = []

    async def wait_if_needed(self):
        """Wait if we've hit the rate limit."""
        now = time.time()
        # Remove old requests outside the window
        self.request_times = [
            t for t in self.request_times if now - t < self.window_seconds
        ]

        if len(self.request_times) >= self.requests_per_window:
            # Need to wait
            oldest = min(self.request_times)
            wait_time = self.window_seconds - (now - oldest) + 1
            console.print(
                f"[yellow]Rate limit reached. Waiting {wait_time:.0f} seconds...[/yellow]"
            )
            await asyncio.sleep(wait_time)

        self.request_times.append(time.time())


def get_bearer_token() -> str | None:
    """Get the Twitter API bearer token from environment."""
    return os.environ.get("TWITTER_BEARER_TOKEN")


async def fetch_tweet(
    tweet_id: str,
    client: httpx.AsyncClient,
    bearer_token: str,
) -> dict | None:
    """Fetch a single tweet by ID."""
    try:
        response = await client.get(
            f"{TWITTER_API_BASE}/tweets/{tweet_id}",
            headers={"Authorization": f"Bearer {bearer_token}"},
            params={
                "expansions": "author_id,referenced_tweets.id",
                "tweet.fields": "created_at,text,author_id,conversation_id,in_reply_to_user_id",
                "user.fields": "username,name",
            },
        )
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 429:
            console.print("[red]Rate limited by Twitter API[/red]")
            return None
        else:
            console.print(f"[yellow]Tweet fetch failed: {response.status_code}[/yellow]")
            return None
    except Exception as e:
        console.print(f"[red]Error fetching tweet: {e}[/red]")
        return None


async def search_replies(
    tweet_id: str,
    conversation_id: str,
    client: httpx.AsyncClient,
    bearer_token: str,
    rate_limiter: RateLimiter,
    pagination_token: str | None = None,
) -> tuple[list[dict], str | None]:
    """Search for replies to a tweet."""
    await rate_limiter.wait_if_needed()

    query = f"conversation_id:{conversation_id}"

    params = {
        "query": query,
        "max_results": 100,
        "expansions": "author_id,referenced_tweets.id",
        "tweet.fields": "created_at,text,author_id,conversation_id,in_reply_to_user_id,referenced_tweets",
        "user.fields": "username,name",
    }

    if pagination_token:
        params["pagination_token"] = pagination_token

    try:
        response = await client.get(
            f"{TWITTER_API_BASE}/tweets/search/recent",
            headers={"Authorization": f"Bearer {bearer_token}"},
            params=params,
        )

        if response.status_code == 200:
            data = response.json()
            tweets = data.get("data", [])
            includes = data.get("includes", {})
            next_token = data.get("meta", {}).get("next_token")

            # Attach user info to tweets
            users = {u["id"]: u for u in includes.get("users", [])}
            for tweet in tweets:
                author_id = tweet.get("author_id")
                if author_id in users:
                    tweet["_user"] = users[author_id]

            return tweets, next_token
        elif response.status_code == 429:
            # Rate limited - wait and retry
            reset_time = response.headers.get("x-rate-limit-reset")
            if reset_time:
                wait_time = int(reset_time) - int(time.time()) + 1
                console.print(
                    f"[yellow]Rate limited. Waiting {wait_time} seconds...[/yellow]"
                )
                await asyncio.sleep(max(wait_time, 1))
                return await search_replies(
                    tweet_id, conversation_id, client, bearer_token, rate_limiter, pagination_token
                )
            return [], None
        else:
            console.print(
                f"[yellow]Search failed: {response.status_code} - {response.text}[/yellow]"
            )
            return [], None
    except Exception as e:
        console.print(f"[red]Error searching replies: {e}[/red]")
        return [], None


async def search_quotes(
    tweet_id: str,
    client: httpx.AsyncClient,
    bearer_token: str,
    rate_limiter: RateLimiter,
    pagination_token: str | None = None,
) -> tuple[list[dict], str | None]:
    """Search for quote tweets of a tweet."""
    await rate_limiter.wait_if_needed()

    params = {
        "max_results": 100,
        "expansions": "author_id",
        "tweet.fields": "created_at,text,author_id",
        "user.fields": "username,name",
    }

    if pagination_token:
        params["pagination_token"] = pagination_token

    try:
        response = await client.get(
            f"{TWITTER_API_BASE}/tweets/{tweet_id}/quote_tweets",
            headers={"Authorization": f"Bearer {bearer_token}"},
            params=params,
        )

        if response.status_code == 200:
            data = response.json()
            tweets = data.get("data", [])
            includes = data.get("includes", {})
            next_token = data.get("meta", {}).get("next_token")

            # Attach user info to tweets
            users = {u["id"]: u for u in includes.get("users", [])}
            for tweet in tweets:
                author_id = tweet.get("author_id")
                if author_id in users:
                    tweet["_user"] = users[author_id]

            return tweets, next_token
        elif response.status_code == 429:
            reset_time = response.headers.get("x-rate-limit-reset")
            if reset_time:
                wait_time = int(reset_time) - int(time.time()) + 1
                console.print(
                    f"[yellow]Rate limited. Waiting {wait_time} seconds...[/yellow]"
                )
                await asyncio.sleep(max(wait_time, 1))
                return await search_quotes(
                    tweet_id, client, bearer_token, rate_limiter, pagination_token
                )
            return [], None
        else:
            console.print(
                f"[yellow]Quote search failed: {response.status_code}[/yellow]"
            )
            return [], None
    except Exception as e:
        console.print(f"[red]Error searching quotes: {e}[/red]")
        return [], None


def parse_tweet_to_response(
    tweet: dict,
    parent_id: str,
    response_type: ResponseType,
) -> Response:
    """Parse a tweet dict into a Response object."""
    user = tweet.get("_user", {})
    username = user.get("username", "unknown")

    created_at_str = tweet.get("created_at", "")
    try:
        created_at = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        created_at = datetime.utcnow()

    tweet_id = tweet.get("id", "")
    url = f"https://twitter.com/{username}/status/{tweet_id}"

    return Response(
        id=tweet_id,
        platform=Platform.TWITTER,
        response_type=response_type,
        author_handle=username,
        author_id=tweet.get("author_id", ""),
        text=tweet.get("text"),
        created_at=created_at,
        parent_id=parent_id,
        url=url,
        raw_data=tweet,
    )


async def scrape_twitter_responses(
    parsed_url: ParsedURL,
    client: httpx.AsyncClient,
    progress_callback=None,
) -> AnalysisResult:
    """Scrape replies and quotes for a Twitter post."""
    bearer_token = get_bearer_token()
    if not bearer_token:
        raise ValueError(
            "TWITTER_BEARER_TOKEN environment variable is required for Twitter scraping"
        )

    rate_limiter = RateLimiter(RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW)
    tweet_id = parsed_url.post_id

    # First, fetch the original tweet to get conversation_id
    if progress_callback:
        progress_callback("Fetching original tweet...")

    await rate_limiter.wait_if_needed()
    original_tweet = await fetch_tweet(tweet_id, client, bearer_token)
    if not original_tweet:
        raise ValueError(f"Could not fetch tweet {tweet_id}")

    tweet_data = original_tweet.get("data", {})
    conversation_id = tweet_data.get("conversation_id", tweet_id)

    responses: list[Response] = []

    # Fetch replies
    if progress_callback:
        progress_callback("Fetching replies...")

    console.print("[blue]Searching for replies...[/blue]")
    pagination_token = None
    page = 0

    while True:
        page += 1
        console.print(f"[dim]Fetching replies page {page}...[/dim]")

        tweets, pagination_token = await search_replies(
            tweet_id, conversation_id, client, bearer_token, rate_limiter, pagination_token
        )

        for tweet in tweets:
            # Skip the original tweet
            if tweet.get("id") == tweet_id:
                continue

            # Determine if it's a direct reply or nested reply
            response = parse_tweet_to_response(tweet, tweet_id, ResponseType.REPLY)
            responses.append(response)

        if not pagination_token:
            break

    console.print(f"[green]Found {len(responses)} replies[/green]")

    # Fetch quote tweets
    if progress_callback:
        progress_callback("Fetching quote tweets...")

    console.print("[blue]Searching for quote tweets...[/blue]")
    pagination_token = None
    quote_count = 0
    page = 0

    while True:
        page += 1
        console.print(f"[dim]Fetching quotes page {page}...[/dim]")

        tweets, pagination_token = await search_quotes(
            tweet_id, client, bearer_token, rate_limiter, pagination_token
        )

        for tweet in tweets:
            response = parse_tweet_to_response(tweet, tweet_id, ResponseType.QUOTE)
            responses.append(response)
            quote_count += 1

        if not pagination_token:
            break

    console.print(f"[green]Found {quote_count} quote tweets[/green]")

    # Count responses by type
    responses_by_type: dict[str, int] = {}
    for response in responses:
        type_name = response.response_type.value
        responses_by_type[type_name] = responses_by_type.get(type_name, 0) + 1

    return AnalysisResult(
        source_url=parsed_url.original_url,
        platform=Platform.TWITTER,
        total_responses=len(responses),
        responses_by_type=responses_by_type,
        responses=responses,
    )
