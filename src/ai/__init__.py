"""AI synthesis module with OpenRouter client and institutional prompt templates."""
from .openrouter_client import generate_ai_trade_summary
from .prompt_templates import build_investment_memo_prompt

__all__ = [
    "generate_ai_trade_summary",
    "build_investment_memo_prompt",
]
