"""Data models for cross-platform interaction analysis."""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Platform(str, Enum):
    """Supported social media platforms."""

    TWITTER = "twitter"
    BLUESKY = "bluesky"


class ResponseType(str, Enum):
    """Types of responses/interactions."""

    REPLY = "reply"              # Direct reply to the target post
    THREAD_REPLY = "thread_reply"  # Reply to someone else in the thread
    QUOTE = "quote"
    REPOST = "repost"
    LIKE = "like"
    MENTION = "mention"


class ParsedURL(BaseModel):
    """Parsed social media URL."""

    platform: Platform
    original_url: str
    username: str
    post_id: str
    at_uri: Optional[str] = None  # For Bluesky AT Protocol URI


class Response(BaseModel):
    """A single response/interaction to a post."""

    id: str
    platform: Platform
    response_type: ResponseType
    author_handle: str
    author_id: str
    text: Optional[str] = None
    created_at: datetime
    parent_id: str
    url: Optional[str] = None
    raw_data: dict = Field(default_factory=dict)
    # Sentiment fields (populated after sentiment analysis)
    sentiment_label: Optional[str] = None  # positive, negative, neutral
    sentiment_score: Optional[float] = None  # confidence score


class AnalysisResult(BaseModel):
    """Result of analyzing responses to a post."""

    source_url: str
    platform: Platform
    total_responses: int
    responses_by_type: dict[str, int]
    responses: list[Response]
    fetched_at: datetime = Field(default_factory=datetime.utcnow)


class SentimentScore(BaseModel):
    """Sentiment analysis score for a response."""

    response_id: str
    label: str  # positive, negative, neutral
    score: float  # confidence score


class SentimentAnalysisResult(BaseModel):
    """Result of sentiment analysis on responses."""

    total_analyzed: int
    sentiment_counts: dict[str, int]
    average_scores: dict[str, float]
    sentiments: list[SentimentScore]
