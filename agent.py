from __future__ import annotations

import os, re
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from tools import (
    estimate_medical_translation,
    build_quote_formula_text,
    build_service_proposal,
)

# =========================
# Parsers
# =========================
def detect_doc_type(t: str) -> Optional[str]:
    t = t.lower()
    if "icf" in t or "同意書" in t:
        return "ICF"
    if "protocol" in t or "試驗計畫書" in t:
        return "Protocol"
    if "ifu" in t:
        return "IFU"
    return None


def detect_lang_pair(t: str) -> Optional[str]:
    t = t.lower()
    if "中翻英" in t or "zh->en" in t:
        return "zh->en"
    if "英翻中" in t or "en->zh" in t:
        return "en->zh"
    return None


def detect_word_count(t: str) -> Optional[int]:
    m = re.search(r"(\d{1,3}(?:,\d{3})*|\d+)\s*(字|words?)", t)
    return int(m.group(1).replace(",", "")) if m else None


def detect_rush(t: str) -> Optional[str]:
    t = t.lower()
    if any(k in t for k in ["不急", "一般", "正常"]):
        return "normal"
    if "12" in t and "小時" in t:
        return "12h"
    if any(k in t for k in ["24", "明天", "急", "rush"]):
        return "24h"
    return None


def wants_quote(t: str) -> bool:
    return any(k in t for k in ["報價", "多少錢", "價格", "費用", "交期"])


def is_yes(t: str) -> bool:
    return any(k in t.lower() for k in ["要", "好", "可以", "需要", "ok"])


def is_no(t: str) -> bool:
    return any(k in t.lower() for k in ["不用", "不要", "先不用", "no"])


def is_thanks(t: str) -> bool:
    return any(k in t.lower() for k in ["謝謝", "感謝", "thanks", "thank you"])


# =========================
# Memory
# =========================
@dataclass
class Memory:
    doc_type: Optional[str] = None
    lang_pair: Optional[str] = None
    word_count: Optional[int] = None
    rush: Optional[str] = None


def is_complete(m: Memory) -> bool:
    return all([m.doc_type, m.lang_pair, m.word_count, m.rush])


def next_missing_question(m: Memory) -> Optional[str]:
    if not m.word_count:
        return "請問大約多少字？"
    if not m.lang_pair:
        return "是中翻英還是英翻中？"
    if not m.doc_type:
        return "請問是 ICF / Protocol 還是 IFU？"
    if not m.rush:
        return "請問什麼時候要呢？"
    return None


# =========================
# Agent
# =========================
class MedTranAgent:
    def __init__(self):
        self.case = Memory()
        self.awaiting_quote = False
        self.quote_offered = False  # ⭐ 關鍵
        self.delivery_email = "a0930591669@gmail.com"

        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
        self.vs = Chroma(persist_directory="./chroma_db", embedding_function=OpenAIEmbeddings())

    def _reset(self):
        self.case = Memory()
        self.awaiting_quote = False
        self.quote_offered = False

    def _update(self, t: str):
        self.case.doc_type = detect_doc_type(t) or self.case.doc_type
        self.case.lang_pair = detect_lang_pair(t) or self.case.lang_pair
        self.case.word_count = detect_word_count(t) or self.case.word_count
        self.case.rush = detect_rush(t) or self.case.rush

    def _quote_text(self) -> str:
        q = estimate_medical_translation(
            self.case.word_count,
            self.case.lang_pair,
            None if self.case.rush == "normal" else self.case.rush,
        )
        return (
            f"📌 估價結果\n"
            f"- 費用：NT${q.estimated_price_twd}\n"
            f"- 交期：{q.estimated_days} 個工作天\n\n"
            f"好的，請將要翻譯的檔案寄至 {self.delivery_email}，感謝您的光臨！"
        )

    def answer(self, user_text: str) -> str:
        t = user_text.strip()

        if t.lower() in ["/reset", "/new"]:
            self._reset()
            return "已重新開始新案件，請提供文件類型、語言方向、字數與是否急件。"

        # ===== waiting for quote decision =====
        if self.awaiting_quote:
            if is_yes(t):
                self.awaiting_quote = False
                return self._quote_text()
            if is_no(t) or is_thanks(t):
                self.awaiting_quote = False
                return "不客氣，感謝您的光臨！"
            # 其他問題 → 直接回答，不再回報價
            return self._fallback(t)

        # ===== general flow =====
        self._update(t)

        # intake gate
        q = next_missing_question(self.case)
        if q:
            return q

        # 主動詢問是否報價（只問一次）
        if is_complete(self.case) and not self.quote_offered:
            self.awaiting_quote = True
            self.quote_offered = True
            return "需要我現在提供報價與交期嗎？（要 / 不用）"

        # 使用者主動提報價
        if wants_quote(t) and is_complete(self.case):
            return self._quote_text()

        # 其他一般問題
        return self._fallback(t)

    def _fallback(self, t: str) -> str:
        docs = self.vs.similarity_search(t, k=3)
        ctx = "\n\n".join(d.page_content for d in docs)
        return self.llm.invoke(
            [
                {"role": "system", "content": "你是翻譯服務客服，回答公司政策與流程。"},
                {"role": "user", "content": ctx + "\n\n" + t},
            ]
        ).content


def main():
    agent = MedTranAgent()
    print("MedTrans 翻譯服務 👋")

    while True:
        t = input("You: ").strip()
        if t.lower() in ["/exit", "exit", "quit"]:
            break
        print("\nAssistant:", agent.answer(t))


if __name__ == "__main__":
    main()
