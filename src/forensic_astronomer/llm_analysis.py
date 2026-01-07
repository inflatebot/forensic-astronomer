"""LLM-based analysis using local LLMs for rich response understanding."""

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
_processor = None  # For vision models
_is_vision_model = False

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
    global _model, _tokenizer, _processor, _is_vision_model

    if _model is not None:
        return _model, _tokenizer, _processor, _is_vision_model

    console.print(f"[cyan]Loading {model_name} with 8-bit quantization...[/cyan]")

    # Check if this is a known vision model
    vision_model_patterns = ["llava", "qwen2-vl", "qwen-vl", "paligemma", "idefics", "cogvlm", "gemma-3"]
    _is_vision_model = any(pattern in model_name.lower() for pattern in vision_model_patterns)

    try:
        from transformers import AutoProcessor, BitsAndBytesConfig
        import torch

        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=False,
        )

        if _is_vision_model:
            console.print(f"[cyan]Detected vision model, loading with processor...[/cyan]")
            from transformers import Gemma3ForConditionalGeneration

            _processor = AutoProcessor.from_pretrained(model_name)
            _tokenizer = _processor.tokenizer if hasattr(_processor, 'tokenizer') else _processor
            _model = Gemma3ForConditionalGeneration.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.bfloat16,
            )
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            _tokenizer = AutoTokenizer.from_pretrained(model_name)
            _model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16,
            )

        # Print diagnostic info
        console.print(f"[green]Model loaded successfully (vision: {_is_vision_model})[/green]")
        console.print(f"[dim]  Vocab size: {len(_tokenizer)}[/dim]")
        console.print(f"[dim]  Model device: {next(_model.parameters()).device}[/dim]")
        if hasattr(_model.config, 'max_position_embeddings'):
            console.print(f"[dim]  Max context: {_model.config.max_position_embeddings}[/dim]")

        return _model, _tokenizer, _processor, _is_vision_model

    except Exception as e:
        console.print(f"[red]Failed to load model: {e}[/red]")
        raise


def load_image(image_path: Path):
    """Load an image for vision model input."""
    try:
        from PIL import Image
        return Image.open(image_path).convert("RGB")
    except ImportError:
        console.print("[yellow]PIL not available for image loading[/yellow]")
        return None
    except Exception as e:
        console.print(f"[yellow]Failed to load image {image_path}: {e}[/yellow]")
        return None


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
    processor=None,
    is_vision_model: bool = False,
    image=None,
) -> Optional[ResponseAnalysis]:
    """Analyze a single response using the LLM."""
    import torch

    if not response.text:
        return None

    # Sanitize text to avoid tokenizer issues
    def sanitize(text: str) -> str:
        if not text:
            return ""
        # Replace problematic characters that can cause tokenizer issues
        return text.encode('utf-8', errors='replace').decode('utf-8')

    prompt = ANALYSIS_PROMPT.format(
        op_handle=sanitize(op_handle),
        op_text=sanitize(op_text) or "[No text available]",
        reply_handle=sanitize(response.author_handle),
        reply_text=sanitize(response.text),
    )

    try:
        if is_vision_model and processor:
            # Gemma 3 vision model
            if image:
                messages = [{"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ]}]
            else:
                # Text-only with vision model
                messages = [{"role": "user", "content": [
                    {"type": "text", "text": prompt},
                ]}]

            inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(model.device)

            input_len = inputs["input_ids"].shape[1]

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.1,
                )

            # Decode only the new tokens
            response_text = processor.decode(
                outputs[0][input_len:],
                skip_special_tokens=True,
            )
        else:
            # Text-only model (non-vision)
            messages = [{"role": "user", "content": prompt}]

            input_ids = tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                add_generation_prompt=True,
            ).to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    pad_token_id=tokenizer.eos_token_id,
                )

            # Decode only the new tokens
            response_text = tokenizer.decode(
                outputs[0][input_ids.shape[1]:],
                skip_special_tokens=True,
            )

        # Log the raw response for debugging
        console.print(f"[dim]@{response.author_handle}: {response.text[:50]}...[/dim]")
        console.print(f"[dim]  → {response_text[:200]}{'...' if len(response_text) > 200 else ''}[/dim]")

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

    except RuntimeError as e:
        error_str = str(e)
        if "CUDA" in error_str or "device-side assert" in error_str:
            console.print(f"[red]CUDA error for {response.id}, resetting...[/red]")
            # Try to recover CUDA state
            try:
                import torch
                torch.cuda.empty_cache()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            except Exception:
                pass
        else:
            console.print(f"[red]Error analyzing {response.id}: {e}[/red]")
        return None
    except Exception as e:
        console.print(f"[red]Error analyzing {response.id}: {e}[/red]")
        return None


def analyze_responses(
    result: AnalysisResult,
    op_text: Optional[str] = None,
    op_image: Optional[Path] = None,
    model_name: str = "google/gemma-3-12b-it",
) -> LLMAnalysisResult:
    """Analyze all responses in an AnalysisResult using the LLM."""
    model, tokenizer, processor, is_vision_model = load_model(model_name)

    # Use stored source post data, with CLI override
    actual_op_text = op_text or result.source_post_text or "[Original post text not available]"
    op_handle = result.source_post_author or "unknown"

    # Fallback: try to extract handle from URL if not stored
    if op_handle == "unknown":
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

    # Load image if provided
    image = None
    if op_image:
        if not is_vision_model:
            console.print(f"[yellow]Warning: --op-image provided but {model_name} is not a vision model.[/yellow]")
            console.print(f"[yellow]Use a vision model like llava-hf/llava-1.5-7b-hf or Qwen/Qwen2-VL-7B-Instruct[/yellow]")
        else:
            image = load_image(op_image)
            if image:
                console.print(f"[green]Loaded image: {op_image}[/green]")

    # Log what we're using
    console.print(f"[dim]OP: @{op_handle}[/dim]")
    console.print(f"[dim]Text: {actual_op_text[:100]}{'...' if len(actual_op_text) > 100 else ''}[/dim]")

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
                op_text=actual_op_text,
                model=model,
                tokenizer=tokenizer,
                processor=processor,
                is_vision_model=is_vision_model,
                image=image,
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
