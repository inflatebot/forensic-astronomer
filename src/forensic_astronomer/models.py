"""Data models for cross-platform interaction analysis."""

from datetime import datetime
from enum import Enum
from typing import Literal, Optional

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
    # Rich LLM analysis fields (populated after LLM analysis)
    llm_is_upset: Optional[bool] = None
    llm_upset_at: Optional[str] = None
    llm_intent: Optional[str] = None
    llm_rhetorical_point: Optional[str] = None
    llm_stance: Optional[str] = None
    llm_reasoning: Optional[str] = None


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


# Rich LLM-based analysis models

UpsetAt = Literal["op", "on_behalf_of_op", "third_party", "situation", "unclear"]
Intent = Literal[
    "emotional_expression",
    "rhetorical_point",
    "question",
    "agreement",
    "disagreement",
    "humor",
    "information",
    "other",
]
Stance = Literal["supportive", "critical", "neutral", "mixed"]


class ResponseAnalysis(BaseModel):
    """Rich analysis of a response using LLM reasoning."""

    response_id: str
    is_upset: bool
    upset_at: Optional[UpsetAt] = None
    intent: Intent
    rhetorical_point: Optional[str] = None  # Summary if making a rhetorical point
    stance_toward_op: Stance
    confidence: float  # Model's confidence in this analysis
    reasoning: str  # Brief explanation of the analysis


class LLMAnalysisResult(BaseModel):
    """Result of LLM-based analysis on responses."""

    model_name: str
    total_analyzed: int
    analyses: list[ResponseAnalysis]
    # Aggregate stats
    upset_count: int
    upset_at_counts: dict[str, int]
    intent_counts: dict[str, int]
    stance_counts: dict[str, int]
