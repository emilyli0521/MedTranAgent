from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from tools import estimate_medical_translation


# =========================
# Parsers
# =========================
def detect_doc_type(t: str) -> Optional[str]:
    tt = t.lower()
    if "icf" in tt or "受試者同意書" in tt or "同意書" in tt:
        return "ICF"
    if "protocol" in tt or "試驗計畫書" in tt:
        return "Protocol"
    if "ifu" in tt or "醫材文件" in tt:
        return "IFU"
    return None


def detect_lang_pair(t: str) -> Optional[str]:
    tt = t.lower()
    if "中翻英" in tt or "zh->en" in tt or "中文翻英文" in tt:
        return "zh->en"
    if "英翻中" in tt or "en->zh" in tt or "英文翻中文" in tt:
        return "en->zh"
    return None


def detect_word_count(t: str) -> Optional[int]:
    """
    支援：
    - 2500字 / 2,500 字
    - 2500 words / 2,500 words
    - 純數字：2500
    - 2500左右 / 2500上下
    """
    tt = t.strip().lower()

    m = re.search(r"(\d{1,3}(?:,\d{3})*|\d+)\s*(字|words?)", tt)
    if m:
        return int(m.group(1).replace(",", ""))

    m2 = re.fullmatch(r"(\d{1,3}(?:,\d{3})*|\d+)\s*(左右|上下|多)?", tt)
    if m2:
        return int(m2.group(1).replace(",", ""))

    return None


def detect_rush(t: str) -> Optional[str]:
    tt = t.lower()
    if any(k in tt for k in ["不急", "非急", "一般", "正常"]):
        return "normal"
    if "12" in tt and "小時" in tt:
        return "12h"
    if ("24" in tt and "小時" in tt) or any(k in tt for k in ["明天", "急件", "急", "rush"]):
        return "24h"
    return None


def wants_quote(t: str) -> bool:
    tt = t.lower()
    return any(k in tt for k in ["報價", "多少錢", "價格", "費用", "交期", "估價", "quote", "多久"])


def asks_non_zh_en_language(t: str) -> Optional[str]:
    tt = t.lower()
    mapping = {
        "日文": "日文", "日語": "日文", "日本語": "日文", "jp": "日文", "ja": "日文",
        "韓文": "韓文", "韓語": "韓文", "kr": "韓文", "ko": "韓文",
        "法文": "法文", "法語": "法文", "fr": "法文",
        "德文": "德文", "德語": "德文", "de": "德文",
        "西班牙文": "西班牙文", "西語": "西班牙文", "es": "西班牙文",
        "義大利文": "義大利文", "it": "義大利文",
        "俄文": "俄文", "俄語": "俄文", "ru": "俄文",
        "葡萄牙文": "葡萄牙文", "pt": "葡萄牙文",
        "泰文": "泰文", "th": "泰文",
        "越南文": "越南文", "vi": "越南文",
    }
    for k, v in mapping.items():
        if k in tt:
            return v
    return None


# =========================
# Intent Routing
# =========================
def _norm(t: str) -> str:
    return re.sub(r"\s+", "", t.strip().lower())


YES_SET = {"要", "好", "可以", "ok", "okay", "yes", "y"}
NO_SET  = {"不用", "不要", "先不用", "no", "n"}

def is_yes(t: str) -> bool:
    tt = _norm(t)
    return tt in YES_SET or tt.startswith(("要報價", "給我報價", "可以報價", "麻煩報價", "請報價"))

def is_no(t: str) -> bool:
    tt = _norm(t)
    return tt in NO_SET

def is_thanks(t: str) -> bool:
    tt = t.lower()
    return any(k in tt for k in ["謝謝", "感謝", "thanks", "thank you"])


def is_noise(t: str) -> bool:
    tt = _norm(t)

    # ✅ 常見肯定/否定不要當 noise（避免「要」被誤判）
    whitelist = {"要", "不用", "不要", "好", "可以", "ok", "okay", "yes", "no", "y", "n"}
    if tt in whitelist:
        return False

    if len(tt) <= 1:
        return True

    noises = {"嗯", "喔", "哈哈", "呵", "不知道", "隨便", "?", "？", "..."}
    return tt in noises


def is_faq_question(t: str) -> bool:
    tt = t.lower()
    faq_patterns = [
        "是什麼", "幹嘛", "用途", "差別", "怎麼", "如何", "可以嗎", "有沒有",
        "有哪些", "哪些", "包含", "內容", "範圍", "類型", "有哪些類型", "有哪些文件",
        "流程", "保密", "nda", "格式", "交付", "檔案", "範本", "範例", "需要什麼",
        "為什麼", "什麼意思",
    ]
    return any(p in tt for p in faq_patterns)


def has_intake_fields(t: str) -> bool:
    return any([
        detect_doc_type(t) is not None,
        detect_lang_pair(t) is not None,
        detect_word_count(t) is not None,
        detect_rush(t) is not None,
    ])


def classify_intent(t: str) -> str:
    """
    - QUOTE_EXPLICIT: 明確問報價/交期
    - QUOTE_INTAKE: 明顯要翻譯/估價、或提供欄位資訊（但不一定問錢）
    - FAQ: 問文件/流程等
    - NOISE: 已讀亂回/太短無法判斷
    """
    if is_noise(t):
        return "NOISE"
    if wants_quote(t):
        return "QUOTE_EXPLICIT"

    tt = t.lower()
    strong_quote_intent = [
        "要翻", "要翻譯", "需要翻", "幫我翻", "想翻",
        "有案件", "有案子", "要估", "估一份", "再估", "報價一下", "想詢價",
        "我要翻譯", "我要報價", "我要估價",
    ]
    if any(k in tt for k in strong_quote_intent):
        return "QUOTE_INTAKE"

    # 有欄位但在問 FAQ → 不要當 intake
    if has_intake_fields(t) and not is_faq_question(t):
        return "QUOTE_INTAKE"

    return "FAQ"


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
    if not m.doc_type:
        return "請問是 受試者同意書(ICF) / 試驗計畫書(Protocol) / 醫材文件(IFU) 哪一種呢？"
    if not m.lang_pair:
        return "目前我們只提供 **中翻英 / 英翻中**，請問要哪一種？"
    if not m.word_count:
        return "請問大約多少字？（可直接回 2500 或 2500字 / 2500 words）"
    if not m.rush:
        return "請問急件或非急件呢？（一般/24小時/12小時）"
    return None


# =========================
# Agent
# =========================
class MedTranAgent:
    def __init__(self):
        self.case = Memory()
        self.awaiting_quote = False
        self.quote_offered = False
        self.mode = "intake"  # "intake" | "chat"

        self.delivery_email = "a0930591669@gmail.com"

        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
        self.vs = Chroma(persist_directory="./chroma_db", embedding_function=OpenAIEmbeddings())

    def _reset_case(self):
        self.case = Memory()
        self.awaiting_quote = False
        self.quote_offered = False

    def _reset_all(self):
        self._reset_case()
        self.mode = "intake"

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

    def _guide_quote(self) -> str:
        return (
            "若要估價，請提供：\n"
            "1) 文件類型（ICF / Protocol / IFU）\n"
            "2) 字數（可直接回 2500）\n"
            "3) 中翻英或英翻中\n"
            "4) 急件或非急件（一般/24小時/12小時）🙂"
        )

    def _language_scope_reply(self, lang_name: str) -> str:
        return (
            f"目前我們僅提供 **中翻英 / 英翻中** 的文件翻譯服務，暫不支援{lang_name}。\n\n"
            + self._guide_quote()
        )

    def answer(self, user_text: str) -> str:
        t = user_text.strip()

        if t.lower() in ["/reset", "/new"]:
            self._reset_all()
            return "已重新開始新案件～\n" + self._guide_quote()

        # ✅ 先處理「要不要報價」的回答（避免被 NOISE/FAQ 分流）
        if self.awaiting_quote:
            if is_yes(t):
                self.awaiting_quote = False
                quote = self._quote_text()
                self.mode = "chat"
                return quote + "\n\n（之後你也可以直接問流程、保密、交付格式等🙂）"

            if is_no(t) or is_thanks(t):
                self.awaiting_quote = False
                self.mode = "chat"
                return "沒問題～若你之後要估價，直接把文件類型/字數/語向/急不急件丟給我就可以🙂"

            return "我收到～你是想要我現在直接報價嗎？（要 / 不用）"

        # ✅ 非中英語言：直接固定回覆
        non_zh_en = asks_non_zh_en_language(t)
        if non_zh_en:
            return self._language_scope_reply(non_zh_en)

        intent = classify_intent(t)

        # NOISE：不出選單，直接用一句引導收斂
        if intent == "NOISE":
            return "我可以協助估價或回答流程問題～\n" + self._guide_quote()

        # FAQ：LLM 回答 + 引導詢價
        if intent == "FAQ":
            ans = self._fallback(t)
            return ans + "\n\n" + self._guide_quote()

        # QUOTE：進詢價流程
        if intent in ("QUOTE_EXPLICIT", "QUOTE_INTAKE"):
            # chat 中再次詢價 → 視為新一單
            if self.mode == "chat":
                self._reset_case()
                self.mode = "intake"

            self._update(t)

            q = next_missing_question(self.case)
            if q:
                return q

            if is_complete(self.case) and not self.quote_offered:
                self.awaiting_quote = True
                self.quote_offered = True
                return "我已整理好需求了，需要我現在提供報價與交期嗎？（要 / 不用）"

            # 若使用者明確問報價且資料齊，也可直接報價
            if wants_quote(t) and is_complete(self.case):
                quote = self._quote_text()
                self.mode = "chat"
                return quote + "\n\n（如果你還有其他文件要估價，直接跟我說「再估一份」或「我要報價」就可以～）"

            return self._guide_quote()

        # 理論上不會到
        return self._fallback(t)

    def _fallback(self, t: str) -> str:
        docs = self.vs.similarity_search(t, k=3)
        ctx = "\n\n".join(d.page_content for d in docs)

        system = (
            "你是翻譯服務客服，回答公司政策與流程、交付方式、保密與格式規範等FAQ。\n"
            "【硬性規則】我們只提供「中翻英 / 英翻中」翻譯服務。\n"
            "若使用者詢問或要求任何其他語言（例如日文/韓文/法文等），你必須明確回覆：目前不支援，並引導回中英互翻。\n"
            "禁止捏造、暗示或推測我們支援其他語言。"
        )

        return self.llm.invoke(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": (ctx + "\n\n" if ctx else "") + t},
            ]
        ).content


def main():
    agent = MedTranAgent()
    print(
        "您好，歡迎光臨 MedTrans，請問您需要什麼翻譯服務? "
        "請您提供：翻譯文件類型 / 字數 / 中翻英或英翻中 / 急件或非急件，"
        "讓小助理更快幫您解決問題喔 🙂\n"
        "我們提供的文件翻譯類型有：受試者同意書 (ICF) / 試驗計畫書 (Protocol) / 醫材文件 (IFU)"
    )

    while True:
        t = input("You: ").strip()
        if t.lower() in ["/exit", "exit", "quit"]:
            break
        print("\nAssistant:", agent.answer(t))


if __name__ == "__main__":
    main()
