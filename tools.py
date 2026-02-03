from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class QuoteResult:
    estimated_price_twd: int
    estimated_days: float
    breakdown: dict


def estimate_medical_translation(
    word_count: int,
    lang_pair: str,          # "zh->en" | "en->zh"
    rush: str | None = None  # None | "24h" | "12h"
) -> QuoteResult:
    """
    簡化版報價工具（可 demo）
    ✅ 新計價規則（你指定的）：
        總價 = 計費字數 × 語言基本費率 × 急件加成
      - 24 小時急件：×1.5
      - 12 小時急件：×2.0

    其他：
    - 最低收費：1000 字
    - 交期估算：一般 2000 字/天；急件強制壓縮（demo 用）
    """

    # 基本費率（TWD / 字）
    base_rates = {
        "zh->en": 3.0,
        "en->zh": 2.5,
    }
    if lang_pair not in base_rates:
        raise ValueError("lang_pair must be 'zh->en' or 'en->zh'")

    # 最低收費（以字數計）
    min_words = 1000
    billable_words = max(int(word_count), min_words)

    # 急件加成
    if rush == "24h":
        rush_multiplier = 1.5
    elif rush == "12h":
        rush_multiplier = 2.0
    elif rush is None:
        rush_multiplier = 1.0
    else:
        raise ValueError("rush must be None, '24h', or '12h'")

    # ✅ 價格：拿掉文件類型加成
    base = billable_words * base_rates[lang_pair]
    total = base * rush_multiplier

    # 交期估算（天）
    # 一般：2000 字/天（demo 用）
    speed = 2000.0
    days = billable_words / speed

    # 急件壓縮（demo）
    if rush == "24h":
        days = min(days, 1.0)
    elif rush == "12h":
        days = min(days, 0.5)

    return QuoteResult(
        estimated_price_twd=int(round(total)),
        estimated_days=round(days, 2),
        breakdown={
            "billable_words": billable_words,
            "base_rate": base_rates[lang_pair],
            "rush_multiplier": rush_multiplier,
        }
    )


def build_quote_formula_text(q: QuoteResult) -> str:
    """
    工具2：把工具1（報價）結果轉成可解釋的「公式＋代入」
    ✅ 新公式（拿掉文件類型加成）：
        總價 = 計費字數 × 語言基本費率 × 急件加成
    """
    bw = q.breakdown["billable_words"]
    base_rate = q.breakdown["base_rate"]
    rush_mul = q.breakdown["rush_multiplier"]

    return (
        "計價公式：總價 = 計費字數 × 語言基本費率 × 急件加成\n"
        f"代入：= {bw} × {base_rate} × {rush_mul}\n"
        f"= 約 NT${q.estimated_price_twd}"
    )


def build_service_proposal(
    q: QuoteResult,
    doc_type: str,
    lang_pair: str,
    word_count: int,
    rush: Optional[str] = None,
) -> str:
    """
    工具3（可選）：用報價結果產出「客服提案摘要」
    doc_type 仍可保留在提案文字中（但不影響計價）
    """
    rush_text = "一般件"
    if rush == "24h":
        rush_text = "急件（24 小時）"
    elif rush == "12h":
        rush_text = "急件（12 小時）"

    return (
        "【服務提案摘要】\n"
        f"- 文件類型：{doc_type}\n"
        f"- 語言方向：{lang_pair}\n"
        f"- 原文字數：{word_count}\n"
        f"- 急件：{rush_text}\n"
        f"- 預估費用：NT${q.estimated_price_twd}\n"
        f"- 預估交期：{q.estimated_days} 個工作天\n"
      )
