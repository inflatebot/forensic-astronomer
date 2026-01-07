"""URL parsers for Twitter and Bluesky links."""

import re
from urllib.parse import urlparse

import httpx

from .models import ParsedURL, Platform


# Twitter URL patterns
TWITTER_PATTERNS = [
    # https://twitter.com/username/status/1234567890
    re.compile(r"(?:https?://)?(?:www\.)?twitter\.com/([^/]+)/status/(\d+)"),
    # https://x.com/username/status/1234567890
    re.compile(r"(?:https?://)?(?:www\.)?x\.com/([^/]+)/status/(\d+)"),
]

# Bluesky URL patterns
BLUESKY_PATTERNS = [
    # https://bsky.app/profile/handle.bsky.social/post/abc123
    re.compile(
        r"(?:https?://)?(?:www\.)?bsky\.app/profile/([^/]+)/post/([a-zA-Z0-9]+)"
    ),
]


def parse_twitter_url(url: str) -> ParsedURL | None:
    """Parse a Twitter/X URL and extract username and post ID."""
    for pattern in TWITTER_PATTERNS:
        match = pattern.match(url)
        if match:
            username, post_id = match.groups()
            return ParsedURL(
                platform=Platform.TWITTER,
                original_url=url,
                username=username,
                post_id=post_id,
            )
    return None


def parse_bluesky_url(url: str) -> ParsedURL | None:
    """Parse a Bluesky URL and extract handle and post ID (rkey)."""
    for pattern in BLUESKY_PATTERNS:
        match = pattern.match(url)
        if match:
            handle, rkey = match.groups()
            return ParsedURL(
                platform=Platform.BLUESKY,
                original_url=url,
                username=handle,
                post_id=rkey,
            )
    return None


def parse_url(url: str) -> ParsedURL | None:
    """Parse a social media URL and return structured data."""
    url = url.strip()

    # Try Twitter first
    result = parse_twitter_url(url)
    if result:
        return result

    # Try Bluesky
    result = parse_bluesky_url(url)
    if result:
        return result

    return None


async def resolve_bluesky_did(handle: str, client: httpx.AsyncClient) -> str | None:
    """Resolve a Bluesky handle to a DID."""
    # Remove @ prefix if present
    handle = handle.lstrip("@")

    # Try the Bluesky API first
    try:
        response = await client.get(
            "https://bsky.social/xrpc/com.atproto.identity.resolveHandle",
            params={"handle": handle},
        )
        if response.status_code == 200:
            data = response.json()
            return data.get("did")
    except Exception:
        pass

    # Fallback: try DNS-based resolution or other PDS
    return None


async def get_bluesky_at_uri(
    parsed: ParsedURL, client: httpx.AsyncClient
) -> str | None:
    """Get the AT Protocol URI for a Bluesky post."""
    if parsed.platform != Platform.BLUESKY:
        return None

    did = await resolve_bluesky_did(parsed.username, client)
    if not did:
        return None

    # Construct the AT URI
    at_uri = f"at://{did}/app.bsky.feed.post/{parsed.post_id}"
    parsed.at_uri = at_uri
    return at_uri
