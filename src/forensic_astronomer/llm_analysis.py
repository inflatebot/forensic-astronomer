"""LLM-based analysis using Gemma 3 for rich response understanding."""

import json
import re
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from .models import (
    AnalysisResult,
    Response,
    ResponseAnalysis,
    LLMAnalysisResult,
)

console = Console()

# Global model cache
_model = None
_tokenizer = None

ANALYSIS_PROMPT = """Analyze this social media reply and respond with JSON only.

Original post by @{op_handle}:
{op_text}

Reply by @{reply_handle}:
{reply_text}

Analyze this reply and output ONLY a JSON object with these fields:
- "is_upset": boolean - Is the replier expressing upset/frustration/anger?
- "upset_at": one of ["op", "on_behalf_of_op", "third_party", "situation", "unclear", null] - Who/what are they upset at? null if not upset
- "intent": one of ["emotional_expression", "rhetorical_point", "question", "agreement", "disagreement", "humor", "information", "other"]
- "rhetorical_point": string or null - If making a rhetorical point, briefly summarize it (1 sentence max)
- "stance_toward_op": one of ["supportive", "critical", "neutral", "mixed"]
- "confidence": float 0-1 - Your confidence in this analysis
- "reasoning": string - 1-2 sentence explanation

JSON only, no other text:"""


def load_model(model_name: str = "google/gemma-3-12b-it"):
    """Load the model with 8-bit quantization."""
    global _model, _tokenizer

    if _model is not None:
        return _model, _tokenizer

    console.print(f"[cyan]Loading {model_name} with 8-bit quantization...[/cyan]")

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        import torch

        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=False,
        )

        _tokenizer = AutoTokenizer.from_pretrained(model_name)
        _model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16,
        )

        console.print(f"[green]Model loaded successfully[/green]")
        return _model, _tokenizer

    except Exception as e:
        console.print(f"[red]Failed to load model: {e}[/red]")
        raise


def parse_llm_response(text: str) -> Optional[dict]:
    """Parse JSON from LLM response, handling common issues."""
    # Try to extract JSON from the response
    text = text.strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find JSON in the response
    json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    # Try to find JSON with nested braces
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    return None


def analyze_single_response(
    response: Response,
    op_handle: str,
    op_text: str,
    model,
    tokenizer,
) -> Optional[ResponseAnalysis]:
    """Analyze a single response using the LLM."""
    import torch

    if not response.text:
        return None

    prompt = ANALYSIS_PROMPT.format(
        op_handle=op_handle,
        op_text=op_text or "[No text available]",
        reply_handle=response.author_handle,
        reply_text=response.text,
    )

    # Format for Gemma 3 chat
    messages = [{"role": "user", "content": prompt}]

    try:
        input_ids = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=300,
                temperature=0.3,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode only the new tokens
        response_text = tokenizer.decode(
            outputs[0][input_ids.shape[1]:],
            skip_special_tokens=True,
        )

        parsed = parse_llm_response(response_text)
        if not parsed:
            console.print(f"[yellow]Failed to parse response for {response.id}[/yellow]")
            return None

        # Validate and normalize the parsed data
        is_upset = bool(parsed.get("is_upset", False))
        upset_at = parsed.get("upset_at")
        if upset_at not in ["op", "on_behalf_of_op", "third_party", "situation", "unclear", None]:
            upset_at = "unclear" if is_upset else None

        intent = parsed.get("intent", "other")
        valid_intents = ["emotional_expression", "rhetorical_point", "question",
                        "agreement", "disagreement", "humor", "information", "other"]
        if intent not in valid_intents:
            intent = "other"

        stance = parsed.get("stance_toward_op", "neutral")
        if stance not in ["supportive", "critical", "neutral", "mixed"]:
            stance = "neutral"

        return ResponseAnalysis(
            response_id=response.id,
            is_upset=is_upset,
            upset_at=upset_at,
            intent=intent,
            rhetorical_point=parsed.get("rhetorical_point"),
            stance_toward_op=stance,
            confidence=min(1.0, max(0.0, float(parsed.get("confidence", 0.5)))),
            reasoning=parsed.get("reasoning", ""),
        )

    except Exception as e:
        console.print(f"[red]Error analyzing {response.id}: {e}[/red]")
        return None


def analyze_responses(
    result: AnalysisResult,
    op_text: Optional[str] = None,
    op_screenshot: Optional[Path] = None,
    model_name: str = "google/gemma-3-12b-it",
) -> LLMAnalysisResult:
    """Analyze all responses in an AnalysisResult using the LLM."""
    model, tokenizer = load_model(model_name)

    # Extract OP handle from source URL
    op_handle = "unknown"
    if "/status/" in result.source_url:
        # Twitter URL
        parts = result.source_url.split("/")
        try:
            op_handle = parts[parts.index("status") - 1]
        except (ValueError, IndexError):
            pass
    elif "bsky.app" in result.source_url or "at://" in result.source_url:
        # Bluesky URL
        parts = result.source_url.split("/")
        for part in parts:
            if part.startswith("did:") or "." in part:
                op_handle = part
                break

    # Filter to responses with text content
    responses_to_analyze = [r for r in result.responses if r.text]

    analyses: list[ResponseAnalysis] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            f"[cyan]Analyzing {len(responses_to_analyze)} responses...",
            total=len(responses_to_analyze),
        )

        for response in responses_to_analyze:
            analysis = analyze_single_response(
                response=response,
                op_handle=op_handle,
                op_text=op_text or "[Original post text not provided]",
                model=model,
                tokenizer=tokenizer,
            )

            if analysis:
                analyses.append(analysis)

                # Write back to response object
                response.llm_is_upset = analysis.is_upset
                response.llm_upset_at = analysis.upset_at
                response.llm_intent = analysis.intent
                response.llm_rhetorical_point = analysis.rhetorical_point
                response.llm_stance = analysis.stance_toward_op
                response.llm_reasoning = analysis.reasoning

            progress.advance(task)

    # Compute aggregate stats
    upset_count = sum(1 for a in analyses if a.is_upset)

    upset_at_counts: dict[str, int] = {}
    for a in analyses:
        if a.upset_at:
            upset_at_counts[a.upset_at] = upset_at_counts.get(a.upset_at, 0) + 1

    intent_counts: dict[str, int] = {}
    for a in analyses:
        intent_counts[a.intent] = intent_counts.get(a.intent, 0) + 1

    stance_counts: dict[str, int] = {}
    for a in analyses:
        stance_counts[a.stance_toward_op] = stance_counts.get(a.stance_toward_op, 0) + 1

    return LLMAnalysisResult(
        model_name=model_name,
        total_analyzed=len(analyses),
        analyses=analyses,
        upset_count=upset_count,
        upset_at_counts=upset_at_counts,
        intent_counts=intent_counts,
        stance_counts=stance_counts,
    )


def print_llm_analysis_summary(result: LLMAnalysisResult):
    """Print a summary of the LLM analysis to the console."""
    console.print()
    console.print(f"[bold]LLM Analysis Summary ({result.model_name})[/bold]")
    console.print(f"Analyzed: {result.total_analyzed} responses")
    console.print()

    # Upset breakdown
    if result.total_analyzed > 0:
        upset_pct = result.upset_count / result.total_analyzed * 100
        console.print(f"[bold]Upset responses:[/bold] {result.upset_count} ({upset_pct:.1f}%)")

        if result.upset_at_counts:
            console.print("[bold]Upset at:[/bold]")
            display_names = {
                "op": "The OP (critical)",
                "on_behalf_of_op": "On OP's behalf (supportive)",
                "third_party": "Third party",
                "situation": "The situation",
                "unclear": "Unclear",
            }
            for target, count in sorted(result.upset_at_counts.items(), key=lambda x: -x[1]):
                pct = count / result.upset_count * 100 if result.upset_count > 0 else 0
                name = display_names.get(target, target)
                console.print(f"  - {name}: {count} ({pct:.1f}%)")

        console.print()

        # Intent breakdown
        console.print("[bold]Intent:[/bold]")
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
        for intent, count in sorted(result.intent_counts.items(), key=lambda x: -x[1]):
            pct = count / result.total_analyzed * 100
            name = intent_names.get(intent, intent)
            console.print(f"  - {name}: {count} ({pct:.1f}%)")

        console.print()

        # Stance breakdown
        console.print("[bold]Stance toward OP:[/bold]")
        for stance, count in sorted(result.stance_counts.items(), key=lambda x: -x[1]):
            pct = count / result.total_analyzed * 100
            console.print(f"  - {stance.title()}: {count} ({pct:.1f}%)")

        # Show rhetorical points if any
        rhetorical = [a for a in result.analyses if a.rhetorical_point]
        if rhetorical:
            console.print()
            console.print(f"[bold]Rhetorical points made ({len(rhetorical)}):[/bold]")
            for a in rhetorical[:10]:  # Show first 10
                console.print(f"  - {a.rhetorical_point}")
            if len(rhetorical) > 10:
                console.print(f"  ... and {len(rhetorical) - 10} more")
