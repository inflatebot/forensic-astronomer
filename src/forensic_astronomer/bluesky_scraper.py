"""Bluesky backlinks scraper using microcosm Constellation API."""

import asyncio
from datetime import datetime
from typing import AsyncIterator
from urllib.parse import quote

import httpx
from rich.console import Console

from .models import AnalysisResult, ParsedURL, Platform, Response, ResponseType
from .parsers import get_bluesky_at_uri

console = Console()

# Microcosm Constellation API
CONSTELLATION_BASE = "https://constellation.microcosm.blue"
BACKLINKS_ENDPOINT = f"{CONSTELLATION_BASE}/xrpc/blue.microcosm.links.getBacklinks"

# Bluesky public API for record details
BSKY_PUBLIC_API = "https://public.api.bsky.app"

# User agent as requested by microcosm
USER_AGENT = "forensic-astronomer/0.1.0 (+https://github.com/inflatebot/forensic-astronomer)"


def determine_response_type(record: dict, target_uri: str) -> ResponseType:
    """Determine the type of response based on the record structure.

    Backlink types in AT Protocol:
    - embed.record.uri: quote posts
    - embed.record.record.uri: quote posts with images/video
    - reply.parent.uri: direct replies to the post
    - reply.root.uri: all replies in the thread (but not direct)
    """
    record_type = record.get("$type", "")

    if record_type == "app.bsky.feed.repost":
        return ResponseType.REPOST

    if record_type == "app.bsky.feed.like":
        return ResponseType.LIKE

    if record_type == "app.bsky.feed.post":
        # Check for reply - direct reply to target
        reply = record.get("reply", {})
        parent_uri = reply.get("parent", {}).get("uri")
        root_uri = reply.get("root", {}).get("uri")

        if parent_uri == target_uri:
            # Direct reply to the target post
            return ResponseType.REPLY
        elif root_uri == target_uri and parent_uri:
            # Reply in the thread but not directly to target
            return ResponseType.REPLY

        # Check for quote post - embed.record.uri
        embed = record.get("embed", {})
        embed_type = embed.get("$type", "")

        if embed_type == "app.bsky.embed.record":
            # Simple quote post
            embed_uri = embed.get("record", {}).get("uri")
            if embed_uri == target_uri:
                return ResponseType.QUOTE
        elif embed_type == "app.bsky.embed.recordWithMedia":
            # Quote post with images/video - embed.record.record.uri
            embed_uri = embed.get("record", {}).get("record", {}).get("uri")
            if embed_uri == target_uri:
                return ResponseType.QUOTE

        # If it's linking to us but not a reply or quote, it's a mention
        return ResponseType.MENTION

    return ResponseType.MENTION


# Source patterns for different link types
# Format: collection:path.to.uri.field
BACKLINK_SOURCES = {
    "reply_parent": "app.bsky.feed.post:reply.parent.uri",      # Direct replies
    "reply_root": "app.bsky.feed.post:reply.root.uri",          # Thread replies
    "quote": "app.bsky.feed.post:embed.record.uri",             # Quote posts
    "quote_media": "app.bsky.feed.post:embed.record.record.uri",# Quote posts with media
    "repost": "app.bsky.feed.repost:subject.uri",               # Reposts
    "like": "app.bsky.feed.like:subject.uri",                   # Likes
}


async def fetch_backlinks(
    at_uri: str,
    source: str,
    client: httpx.AsyncClient,
    cursor: str | None = None,
    limit: int = 100,
) -> tuple[list[dict], str | None]:
    """Fetch backlinks for a given AT URI from Constellation."""
    params = {
        "subject": at_uri,
        "source": source,
        "limit": limit,
    }
    if cursor:
        params["cursor"] = cursor

    response = await client.get(
        BACKLINKS_ENDPOINT,
        params=params,
        headers={"User-Agent": USER_AGENT},
    )

    if response.status_code != 200:
        console.print(f"[red]API Error {response.status_code}: {response.text}[/red]")
        response.raise_for_status()

    data = response.json()

    records = data.get("records", [])
    next_cursor = data.get("cursor")

    return records, next_cursor


async def fetch_record(
    at_uri: str,
    client: httpx.AsyncClient,
) -> dict | None:
    """Fetch a record from the Bluesky public API."""
    # Parse the AT URI: at://did/collection/rkey
    parts = at_uri.replace("at://", "").split("/")
    if len(parts) < 3:
        return None

    did = parts[0]
    collection = parts[1]
    rkey = parts[2]

    try:
        response = await client.get(
            f"{BSKY_PUBLIC_API}/xrpc/com.atproto.repo.getRecord",
            params={
                "repo": did,
                "collection": collection,
                "rkey": rkey,
            },
            headers={"User-Agent": USER_AGENT},
        )
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        console.print(f"[yellow]Warning: Could not fetch record {at_uri}: {e}[/yellow]")

    return None


async def resolve_handle(did: str, client: httpx.AsyncClient) -> str:
    """Resolve a DID to a handle."""
    try:
        response = await client.get(
            f"{BSKY_PUBLIC_API}/xrpc/com.atproto.identity.resolveHandle",
            params={"handle": did},
            headers={"User-Agent": USER_AGENT},
        )
        if response.status_code == 200:
            return response.json().get("handle", did)
    except Exception:
        pass

    # Try identity resolution
    try:
        response = await client.get(
            f"https://plc.directory/{did}",
            headers={"User-Agent": USER_AGENT},
        )
        if response.status_code == 200:
            data = response.json()
            # Extract handle from alsoKnownAs
            also_known_as = data.get("alsoKnownAs", [])
            for aka in also_known_as:
                if aka.startswith("at://"):
                    return aka.replace("at://", "")
    except Exception:
        pass

    return did


async def parse_backlink_to_response(
    record_info: dict,
    target_uri: str,
    client: httpx.AsyncClient,
) -> Response | None:
    """Parse a backlink record into a Response object.

    record_info has format: {did, collection, rkey}
    """
    did = record_info.get("did")
    collection = record_info.get("collection")
    rkey = record_info.get("rkey")

    if not all([did, collection, rkey]):
        return None

    # Construct the AT URI
    source_uri = f"at://{did}/{collection}/{rkey}"

    # Fetch the full record to get details
    record_data = await fetch_record(source_uri, client)
    if not record_data:
        return None

    record = record_data.get("value", {})

    # Get the handle
    handle = await resolve_handle(did, client)

    # Parse the creation time
    created_at_str = record.get("createdAt", "")
    try:
        created_at = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        created_at = datetime.utcnow()

    # Determine response type
    response_type = determine_response_type(record, target_uri)

    # Construct the bsky.app URL
    url = f"https://bsky.app/profile/{handle}/post/{rkey}"

    return Response(
        id=source_uri,
        platform=Platform.BLUESKY,
        response_type=response_type,
        author_handle=handle,
        author_id=did,
        text=record.get("text"),
        created_at=created_at,
        parent_id=target_uri,
        url=url,
        raw_data=record_data,
    )


async def scrape_bluesky_backlinks(
    parsed_url: ParsedURL,
    client: httpx.AsyncClient,
    progress_callback=None,
    include_likes: bool = False,
) -> AnalysisResult:
    """Scrape all backlinks for a Bluesky post."""
    # Get the AT URI
    at_uri = await get_bluesky_at_uri(parsed_url, client)
    if not at_uri:
        raise ValueError(f"Could not resolve AT URI for {parsed_url.original_url}")

    console.print(f"[blue]Resolved AT URI: {at_uri}[/blue]")

    responses: list[Response] = []
    seen_ids: set[str] = set()  # Track seen response IDs to avoid duplicates

    # Fetch backlinks for each source type
    for source_name, source_pattern in BACKLINK_SOURCES.items():
        # Skip likes unless explicitly requested
        if source_name == "like" and not include_likes:
            console.print(f"[dim]Skipping likes (use --include-likes to include)[/dim]")
            continue
        console.print(f"[dim]Fetching {source_name} backlinks...[/dim]")
        cursor = None
        page = 0
        source_count = 0

        while True:
            page += 1
            if progress_callback:
                progress_callback(f"Fetching {source_name} page {page}...")

            try:
                links, cursor = await fetch_backlinks(at_uri, source_pattern, client, cursor)
            except Exception as e:
                console.print(f"[yellow]Warning fetching {source_name}: {e}[/yellow]")
                break

            if not links:
                break

            source_count += len(links)

            # Process links concurrently in batches
            batch_size = 10
            for i in range(0, len(links), batch_size):
                batch = links[i : i + batch_size]
                tasks = [
                    parse_backlink_to_response(link, at_uri, client) for link in batch
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                for result in results:
                    if isinstance(result, Response):
                        # Avoid duplicates (same post might appear in multiple source types)
                        if result.id not in seen_ids:
                            seen_ids.add(result.id)
                            responses.append(result)
                    elif isinstance(result, Exception):
                        console.print(f"[yellow]Warning: {result}[/yellow]")

                # Small delay to be polite to the API
                await asyncio.sleep(0.1)

            if not cursor:
                break

        if source_count > 0:
            console.print(f"[dim]  Found {source_count} {source_name} links[/dim]")

    console.print(f"[green]Total unique responses: {len(responses)}[/green]")

    # Count responses by type
    responses_by_type: dict[str, int] = {}
    for response in responses:
        type_name = response.response_type.value
        responses_by_type[type_name] = responses_by_type.get(type_name, 0) + 1

    return AnalysisResult(
        source_url=parsed_url.original_url,
        platform=Platform.BLUESKY,
        total_responses=len(responses),
        responses_by_type=responses_by_type,
        responses=responses,
    )
