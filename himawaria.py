"""Himawaria: 会話記憶検索型の人工無能。

過去の会話ログ(記憶)から、入力に最も近い発言の「次の発言」を返信として選ぶ。

構成:
    tokenize_ngrams      -- 文字n-gram分割 (TF-IDF用)
    NgramTokenizer       -- 結合度ベースの簡易トークナイザー (単語置換用)
    SemanticSpace        -- PPMI+SVDの意味空間 (単語演算・未知語のサブワード補完・SIF文ベクトル)。
                            単語置換の類似判定・意味検索・意味スコアのすべてをここが担う
    DenseIndex           -- 記憶の全発言の意味ベクトル行列 (意味検索用、増分更新対応)
    TfidfIndex           -- 転置索引つきTF-IDF索引 (増分更新対応)
    Himawaria            -- 学習・検索・応答の本体
"""
from __future__ import annotations

import difflib
import heapq
import json
import logging
import math
import os
import pickle
import random
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from rapidfuzz.distance import Levenshtein

try:  # 無くても動く (SemanticSpace=密ベクトルだけが無効になる)
    import numpy as np
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import svds
except ImportError:  # pragma: no cover
    np = None

logger = logging.getLogger(__name__)


def setup_logging(
    level: int = logging.INFO,
    log_file: Path | str | None = None,
    max_bytes: int = 5_000_000,
    backup_count: int = 3,
) -> None:
    """ログの出力先を設定する。コンソールに出し、log_file を指定するとファイルにも残す。

    ファイルは max_bytes ごとにローテーションし、backup_count 世代まで保持する。
    何度呼んでも二重出力にならない。
    """
    from logging.handlers import RotatingFileHandler

    root = logging.getLogger(__name__)
    root.setLevel(level)
    root.propagate = False
    for handler in list(root.handlers):  # 再設定時に重複させない
        root.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root.addHandler(console)

    if log_file is not None:
        file_handler = RotatingFileHandler(
            log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf8"
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

# ---------------------------------------------------------------------------
# 定数
# ---------------------------------------------------------------------------
SYSTEM_MARKERS = frozenset({"!good", "!bad"})
NOISE_PREFIXES = ("!command", "通知:")  # 会話ではない行 (コマンド・通知)。学習から除く
BOT_SPEAKERS = frozenset({"!", "!output", "bot", "init"})
BOT_OUTPUT_SPEAKERS = frozenset({"!", "!output"})
RESERVED_SPEAKERS = frozenset({"!input", "!output", "!system"})

MAX_SENTENCES = 1_600_000
LOG_LIMIT = 9
TOKENIZER_FILE = "tokenizer.json"
LEGACY_TOKENIZER_FILE = "tokenizer.model"  # 旧pickle形式 (読み込みのみ対応)

SEMANTIC_DROP_THRESHOLD = 0.4
REPLACE_CHAR_SIMILARITY = 0.85
REPLACE_SEMANTIC_SIMILARITY = 0.75
DUPLICATE_SIMILARITY = 0.85

_REPEAT_RE = re.compile(r"(.)\1{2,}")
_DIGIT_SPLIT_RE = re.compile(r"(\d+)")
# 漢字 / ひらがな / カタカナ(長音含む) / 英数字 の連続。これ以外 (句読点・記号・空白) は語に含めない
_SCRIPT_RUN_RE = re.compile(r"[一-鿿々〆]+|[ぁ-ゟ]+|[ァ-ヿ]+|[A-Za-z0-9Ａ-Ｚａ-ｚ０-９]+")


@dataclass(frozen=True)
class SearchConfig:
    """検索・スコアリングの調整値。"""

    min_tfidf_candidate_score: float = 0.01
    good_tag_multiplier: float = 1.15
    text_score_weight: float = 0.62
    context_score_weight: float = 0.0  # 0なら文脈スコアは計算自体をスキップ
    history_score_weight: float = 0.10
    good_tag_bonus: float = 0.04
    chara_style_weight: float = 0.02
    near_tie_threshold: float = 0.035
    near_tie_pool_size: int = 3
    history_decay_weights: tuple[float, ...] = (1.0, 0.5, 0.25, 0.125, 0.0625)
    history_lookback: int = 100  # 履歴スコアで遡る記憶エントリ数の上限
    min_query_matches: int = 2
    min_coverage: float = 0.20  # 異なるトークンが4個以上のときだけ適用
    min_text_score: float = 0.01
    max_scored_candidates: int = 300  # TF-IDF上位だけを詳細採点する (0以下で無制限)
    semantic_score_weight: float = 0.25  # 入力と候補の「意味の近さ」(文ベクトルのコサイン) の重み。0で無効
    dense_candidates: int = 50  # 意味ベクトルの近さだけで拾う候補の上限。0で無効
    dense_min_sim: float = 0.55  # 意味検索で候補に入れる最低コサイン。返信してよいかの意味側の合格ラインも兼ねる
    semantic_score_floor: float = 0.3  # 意味スコアの下限。これ以下のコサインは加点0、1.0で満点 (候補間の相対ではなく絶対尺度)


# ---------------------------------------------------------------------------
# ユーティリティ
# ---------------------------------------------------------------------------
def tokenize_ngrams(text: str, n: int = 2) -> list[str]:
    """文字n-gram分割。連続文字の圧縮と数字列の保護を行う。"""
    if not text:
        return []

    # 同じ文字の3回以上の連続は2文字に圧縮
    text = _REPEAT_RE.sub(r"\1\1", text)
    tokens: list[str] = []
    for part in _DIGIT_SPLIT_RE.split(text):
        if not part:
            continue
        if part.isdigit() or len(part) < n:
            tokens.append(part)
        else:
            tokens.extend(part[i : i + n] for i in range(len(part) - n + 1))
    return tokens


def _sparse_dot(a: dict[str, float], b: dict[str, float]) -> float:
    """疎ベクトルの内積 (小さい方を走査する)。"""
    if len(a) > len(b):
        a, b = b, a
    return sum(v * b[k] for k, v in a.items() if k in b)


def _normalize_vector(vector: dict[str, float]) -> dict[str, float]:
    norm = math.sqrt(sum(v * v for v in vector.values()))
    return {k: v / norm for k, v in vector.items()} if norm else {}


def _normalize_scores(values: list[float]) -> list[float]:
    """min-max正規化。全て同値なら全部0。"""
    if not values:
        return []
    low, high = min(values), max(values)
    if high - low <= 1e-12:
        return [0.0] * len(values)
    span = high - low
    return [(v - low) / span for v in values]


def _entry_text(entry) -> str:
    return str(entry[0]) if entry and entry[0] is not None else ""


def _is_trainable(text, speaker) -> bool:
    """トークナイザー・意味ベクトルの学習に使う発言か。

    !good/!bad、システム行、コマンド行、通知行は会話ではないので除く。
    """
    if not isinstance(text, str) or not text.strip():
        return False
    if text in SYSTEM_MARKERS or "!system" in str(speaker):
        return False
    return not text.startswith(NOISE_PREFIXES)


def _is_system_or_marker(entry) -> bool:
    text, speaker = entry[0], str(entry[1])
    return "!system" in speaker or text in SYSTEM_MARKERS or speaker == "!"


def _is_user_entry(entry) -> bool:
    """人間ユーザーの発言か (bot・システム・直接学習分を除く)。"""
    if _is_system_or_marker(entry):
        return False
    speaker = str(entry[1])
    return speaker not in BOT_SPEAKERS and not speaker.startswith("!input-")


# ---------------------------------------------------------------------------
# トークナイザー
# ---------------------------------------------------------------------------
class NgramTokenizer:
    """文字の結合度(PMI)を学習し、単語置換処理で使うトークナイザー。

    - 登録単語 (話者名など) は周りの文字に関係なく、必ず1トークンとして切り出す。
    - それ以外は、隣り合う2文字のPMIが閾値未満の所、または前後より明確に
      落ち込んでいる所で区切る。(閾値だけだと、記憶にある文はほぼ全部が
      高PMIになり、文全体が1トークンになってしまう)
    """

    DIP_MARGIN = 1.0  # 前後のPMIよりこれ以上低ければ「語の切れ目」とみなす
    MIN_FORCED_LEN = 2  # 1文字の登録語は本文を細切れにしすぎるので受け付けない

    def __init__(self):
        self.char_counts: Counter = Counter()
        self.bigram_counts: Counter = Counter()
        self.forced_words: set[str] = set()
        self._forced_index: dict[str, list[int]] = {}  # 先頭文字 -> 長い順の語長
        self._total_chars = 0
        self._total_bigrams = 0

    def _recount(self):
        """PMIの分母になる合計値を数え直す (毎回sum()すると非常に遅い)"""
        self._total_chars = sum(self.char_counts.values())
        self._total_bigrams = sum(self.bigram_counts.values())

    def train(self, corpus: Iterable[str]):
        """1文字と2文字の出現頻度をカウント"""
        for sentence in corpus:
            if not sentence or not isinstance(sentence, str):
                continue
            self.char_counts.update(sentence)
            self.bigram_counts.update(
                sentence[i : i + 2] for i in range(len(sentence) - 1)
            )
            self._total_chars += len(sentence)
            self._total_bigrams += len(sentence) - 1

    def register_words(self, words: Iterable[str]):
        """分割してほしくない単語 (話者名など) を登録する"""
        changed = False
        for word in words:
            if not word or len(word) < self.MIN_FORCED_LEN or word in self.forced_words:
                continue
            self.forced_words.add(word)
            changed = True
        if changed:
            index: dict[str, set[int]] = {}
            for word in self.forced_words:
                index.setdefault(word[0], set()).add(len(word))
            self._forced_index = {c: sorted(lens, reverse=True) for c, lens in index.items()}

    def _cohesion(self, c1: str, c2: str) -> float:
        """2文字の結合度 (条件付き確率 P(c2|c1))"""
        char_count = self.char_counts.get(c1, 0)
        bigram_count = self.bigram_counts.get(c1 + c2, 0)
        return bigram_count / char_count if char_count and bigram_count else 0.0

    def _pmi(self, c1: str, c2: str) -> float:
        """2文字の相互情報量 (PMI) を計算。文字種に依存せず、統計的な結合度を測る。"""
        bigram_count = self.bigram_counts.get(c1 + c2, 0)
        if bigram_count == 0 or not self._total_chars or not self._total_bigrams:
            return 0.0

        p_x = self.char_counts.get(c1, 0) / self._total_chars
        p_y = self.char_counts.get(c2, 0) / self._total_chars
        p_xy = bigram_count / self._total_bigrams

        # PMI = log2( P(x,y) / (P(x)*P(y)) )
        return math.log2(p_xy / (p_x * p_y + 1e-10))

    def _segments(
        self, text: str, index: dict[str, list[int]], words: set[str]
    ) -> list[tuple[str, bool]]:
        """登録単語(最長一致)とそれ以外に分ける。 [(文字列, 登録単語か)]"""
        segments: list[tuple[str, bool]] = []
        start = 0  # 未確定の通常テキストの開始位置
        i = 0
        n = len(text)
        while i < n:
            matched = 0
            for length in index.get(text[i], ()):
                if text[i : i + length] in words:
                    matched = length
                    break
            if matched:
                if start < i:
                    segments.append((text[start:i], False))
                segments.append((text[i : i + matched], True))
                i += matched
                start = i
            else:
                i += 1
        if start < n:
            segments.append((text[start:], False))
        return segments

    def _split_by_pmi(self, text: str, drop_threshold: float) -> list[str]:
        if len(text) <= 1:
            return [text] if text else []

        scores = [self._pmi(a, b) for a, b in zip(text, text[1:])]
        last = len(scores) - 1
        words: list[str] = []
        start = 0
        for i, score in enumerate(scores):  # i は text[i] と text[i+1] の境目
            cut = score < drop_threshold
            if not cut and 0 < i < last:
                # 両隣より明確に結合が弱い所も語の切れ目とする
                cut = score <= min(scores[i - 1], scores[i + 1]) - self.DIP_MARGIN
            if cut:
                words.append(text[start : i + 1])
                start = i + 1
        words.append(text[start:])
        return words

    def tokenize(
        self,
        text: str,
        drop_threshold: float = 1.0,
        extra_words: Iterable[str] | None = None,
    ) -> list[str]:
        """登録単語を先に切り出し、残りをPMI（統計的結びつき）でtokenizeする。

        extra_words はこの呼び出しだけ有効な追加の登録単語。置換したい語が
        助詞とくっついて1トークンになっていても、必ず単独で切り出せる。
        """
        if not text:
            return []

        index, words = self._forced_index, self.forced_words
        extras = {w for w in (extra_words or ()) if w and w in text}
        if extras:
            words = words | extras
            lengths: dict[str, set[int]] = {c: set(ls) for c, ls in index.items()}
            for word in extras:
                lengths.setdefault(word[0], set()).add(len(word))
            index = {c: sorted(ls, reverse=True) for c, ls in lengths.items()}
        if not index:
            return self._split_by_pmi(text, drop_threshold)

        tokens: list[str] = []
        for segment, forced in self._segments(text, index, words):
            if forced:
                tokens.append(segment)
            else:
                tokens.extend(self._split_by_pmi(segment, drop_threshold))
        return tokens

    def tokenize_words(self, text: str, hira_threshold: float = 0.4) -> list[str]:
        """意味ベクトル用の語分割。文字種の切れ目を使うので、語の途中で切れにくい。

        漢字の連続・カタカナの連続・英数字の連続はそのまま1語にし、ひらがなの連続だけを
        PMIで割る (「映画音楽ばっかり聴いてる」→ 映画音楽 / ばっ / かり / 聴 / いてる)。
        句読点や記号は捨てる。登録単語 (話者名など) は先に1語として切り出す。
        """
        if not text:
            return []
        segments = (
            self._segments(text, self._forced_index, self.forced_words)
            if self._forced_index
            else [(text, False)]
        )
        tokens: list[str] = []
        for segment, forced in segments:
            if forced:
                tokens.append(segment)
                continue
            for match in _SCRIPT_RUN_RE.finditer(segment):
                run = match.group()
                if "ぁ" <= run[0] <= "ゟ":
                    tokens.extend(self._split_by_pmi(run, hira_threshold))
                else:
                    tokens.append(run)
        return tokens

    def save(self, file_path: Path | str):
        data = {
            "char_counts": dict(self.char_counts),
            "bigram_counts": dict(self.bigram_counts),
            "forced_words": sorted(self.forced_words),
        }
        with open(file_path, "w", encoding="utf8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4, sort_keys=True, separators=(',', ': '))

    def load(self, file_path: Path | str):
        with open(file_path, "r", encoding="utf8") as f:
            data = json.load(f)
        self.char_counts = Counter(data.get("char_counts", {}))
        self.bigram_counts = Counter(data.get("bigram_counts", {}))
        self._recount()
        self.forced_words.clear()
        self._forced_index = {}
        self.register_words(data.get("forced_words", []))
        logger.info("トークナイザーモデルを読み込みました: %s", file_path)

    def load_legacy_pickle(self, file_path: Path | str):
        """旧形式(pickle)の読み込み。自分で作った信頼できるファイルにのみ使うこと。"""
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        self.char_counts = Counter(data.get("char_counts", {}))
        self.bigram_counts = Counter(data.get("bigram_counts", {}))
        self._recount()
        logger.info("旧形式のトークナイザーモデルを読み込みました: %s", file_path)


# ---------------------------------------------------------------------------
# 意味空間 (PPMI → SVD の密ベクトル + サブワード補完 + SIF文ベクトル)
# ---------------------------------------------------------------------------
class SemanticSpace:
    """記憶だけから作る意味空間。単語・文・検索・置換の「意味の近さ」はすべてここが担当する。

    旧設計は「疎PPMI (置換用)」と「密SVD (検索用)」が別々に存在し、語分割も別、
    未知語は完全に無視、という二重構造だった。ここでは1つに統合し、次の流れで作る。

    1. 語分割   : NgramTokenizer.tokenize_words (文字種の切れ目 + ひらがなだけPMI)
    2. 共起     : 同じ発言内 (距離で重み付け 1/距離) + 直前の発言との間 (弱く)
    3. PPMI     : 文脈語の確率を cds 乗で平滑化。shift で弱い関連を切り捨て
    4. SVD      : dim 次元に圧縮 (語ベクトル = U·√S)
    5. 後処理   : 平均と上位主成分を除去 (どの語にも似てしまう成分を消す) → 単位長
    6. サブワード: 文字2-3gramごとの「平均ベクトル」を作り、未知語 (誤字・新語・活用違い) も
                  含まれる文字列から近似ベクトルを合成する (fastText的な補完)
    7. 文ベクトル: SIF重み付き平均 → 全発言に共通する第1主成分を除去 → 単位長
    """

    _CHUNK = 20_000  # 文ベクトル化を一度に処理する発言数 (メモリ節約)

    def __init__(
        self,
        dim: int = 64,
        window: int = 5,
        min_count: int = 2,
        cds: float = 0.75,
        shift: float = 1.0,
        turn_weight: float = 0.3,
        max_repeat: int = 2,
        max_vocab: int = 30_000,
        hira_threshold: float = 0.4,
        sif_a: float = 1e-3,
        remove_components: int = 1,
        sub_ngrams: tuple[int, ...] = (2, 3),
        sub_max_df_ratio: float = 0.02,
        oov_weight: float = 0.5,
    ):
        self.dim = dim
        self.window = max(1, min(window, 8))
        self.min_count = min_count
        self.cds = cds
        self.shift = max(shift, 1e-9)  # 1.0で無効。大きいほど弱い共起を捨てる
        self.turn_weight = turn_weight
        self.max_repeat = max_repeat
        self.max_vocab = max_vocab
        self.hira_threshold = hira_threshold
        self.sif_a = sif_a
        self.remove_components = remove_components
        self.sub_ngrams = tuple(sub_ngrams)
        self.sub_max_df_ratio = sub_max_df_ratio
        self.oov_weight = oov_weight  # 未知語 (サブワード補完) を文ベクトルに混ぜる強さ。0で無効

        self.ready = False
        self.version = 0  # 学習し直すたびに増える。密索引が作り直しの要否を判断する
        self.tokenizer: NgramTokenizer | None = None
        self.words: list[str] = []
        self.vocab: dict[str, int] = {}
        self.embeddings = None  # (語数, dim) 各行は単位長
        self.counts = None
        self.doc_freq = None
        self.doc_total = 0
        self._lengths = None
        self._weights = None  # SIF重み (頻出語ほど軽い)
        self._oov_sif = 1.0
        self._sent_pc = None  # 文ベクトルから引く第1主成分
        self._ngram_ids: dict[str, int] = {}
        self._ngram_emb = None
        self._ngram_idf = None
        self._sentence_cache: dict[str, object] = {}

    # ------------------------------------------------------------------
    # 語分割
    # ------------------------------------------------------------------
    @staticmethod
    def _keep_token(token: str) -> bool:
        """助詞・語尾の1文字 (は・が・て…) は捨てる。漢字1文字 (猫・車) は語として残す。"""
        return len(token) > 1 or "一" <= token <= "鿿"

    def _tokens(self, text: str) -> list[str]:
        if self.tokenizer is None:
            return []
        return [t for t in self.tokenizer.tokenize_words(text, self.hira_threshold) if self._keep_token(t)]

    @staticmethod
    def _usable_text(text) -> bool:
        return (
            isinstance(text, str)
            and bool(text.strip())
            and text not in SYSTEM_MARKERS
            and not text.startswith(NOISE_PREFIXES)
        )

    # ------------------------------------------------------------------
    # 学習
    # ------------------------------------------------------------------
    def train_from_memory(self, sentence_pairs: Sequence, tokenizer: NgramTokenizer):
        """Himawaria から呼ばれる入口。失敗しても本体は止めない。"""
        try:
            self.fit(sentence_pairs, forced_words=tokenizer.forced_words)
        except Exception:
            self.ready = False
            logger.exception("意味空間の構築に失敗しました")

    def fit(self, sentence_pairs: Sequence, forced_words: Iterable[str] = ()) -> None:
        """memory["sentence"] の [発言, 話者] ペアから学習し直す"""
        self.ready = False
        self.version += 1
        self._sentence_cache = {}
        if np is None:
            logger.warning("numpy/scipy が無いので意味空間は作りません (意味検索・意味類似は無効)")
            return
        started = time.perf_counter()

        texts: list[str] = []
        repeats: Counter = Counter()
        for entry in sentence_pairs:
            if not isinstance(entry, (list, tuple)) or len(entry) < 2:
                continue
            text, speaker = entry[0], entry[1]
            if not _is_trainable(text, speaker):
                continue
            repeats[text] += 1
            if repeats[text] <= self.max_repeat:
                texts.append(text)
        if len(texts) < 3:
            return

        tokenizer = NgramTokenizer()
        tokenizer.train(texts)
        tokenizer.register_words(forced_words)
        self.tokenizer = tokenizer
        sequences = [self._tokens(text) for text in texts]

        word_counts: Counter = Counter()
        doc_freq: Counter = Counter()
        for seq in sequences:
            word_counts.update(seq)
            doc_freq.update(set(seq))
        words = [w for w, c in word_counts.most_common(self.max_vocab) if c >= self.min_count]
        if len(words) < 5:
            logger.info("意味空間: 語彙が少なすぎるので作りません (%d語)", len(words))
            return
        vocab = {w: i for i, w in enumerate(words)}

        cooc = self._count_cooccurrence(sequences, vocab)
        if not cooc:
            return

        size = len(words)
        keys = np.array(list(cooc.keys()), dtype=np.int64)
        values = np.array(list(cooc.values()), dtype=np.float64)
        matrix = coo_matrix((values, (keys[:, 0], keys[:, 1])), shape=(size, size)).tocsr()

        # PPMI (文脈語の確率を cds 乗で平滑化、shift で弱い関連を切る)
        total = matrix.sum()
        row_sum = np.asarray(matrix.sum(axis=1)).ravel()
        smoothed = row_sum ** self.cds
        context_prob = smoothed / smoothed.sum()
        coo = matrix.tocoo()
        with np.errstate(divide="ignore", invalid="ignore"):
            pmi = np.log((coo.data / total) / ((row_sum[coo.row] / total) * context_prob[coo.col]))
        pmi = np.where(np.isfinite(pmi), np.maximum(pmi - math.log(self.shift), 0.0), 0.0)
        ppmi = coo_matrix((pmi, (coo.row, coo.col)), shape=(size, size)).tocsr()
        ppmi.eliminate_zeros()

        # SVD で圧縮
        k = min(self.dim, size - 1)
        if k < 2 or ppmi.nnz == 0:
            return
        try:
            u, s, _ = svds(ppmi, k=k, random_state=0)
        except Exception:  # ARPACKが収束しないとき。小さければ素直に全部分解する
            if size > 4000:
                logger.exception("SVDに失敗しました")
                return
            u, s, _ = np.linalg.svd(ppmi.toarray(), full_matrices=False)
            u, s = u[:, :k], s[:k]
        emb = self._postprocess(u * np.sqrt(np.maximum(s, 0.0)))

        counts = np.array([word_counts[w] for w in words], dtype=np.float64)
        self.words = words
        self.vocab = vocab
        self.embeddings = emb.astype(np.float32)
        self.counts = counts
        self.doc_freq = np.array([doc_freq[w] for w in words], dtype=np.float64)
        self.doc_total = len(sequences)
        self._lengths = np.array([len(w) for w in words])
        probs = counts / counts.sum()
        self._weights = self.sif_a / (self.sif_a + probs)
        self._oov_sif = self.sif_a / (self.sif_a + 1.0 / counts.sum())  # 未知語は「ごく稀」扱い
        self._build_subword_table(counts)
        self._fit_sentence_component(sequences)
        self.ready = True
        logger.info(
            "意味空間を構築しました。発言 %d / 語彙 %d / %d次元 / サブワード %d (%.2f秒)",
            len(texts), size, emb.shape[1], len(self._ngram_ids), time.perf_counter() - started,
        )

    def _count_cooccurrence(self, sequences: list[list[str]], vocab: dict[str, int]) -> Counter:
        """共起: 同じ発言内 (距離で重み付け) と、直前の発言との間 (弱く)"""
        cooc: Counter = Counter()
        prev: list[int] = []
        for seq in sequences:
            ids = [vocab[t] for t in seq if t in vocab]
            if not ids:
                continue
            n = len(ids)
            for a in range(n):
                for b in range(a + 1, min(n, a + self.window + 1)):
                    if ids[a] == ids[b]:
                        continue
                    weight = 1.0 / (b - a)
                    cooc[(ids[a], ids[b])] += weight
                    cooc[(ids[b], ids[a])] += weight
            current = list(dict.fromkeys(ids))[:10]
            if prev and self.turn_weight:
                for p in prev:
                    for c in current:
                        if p != c:
                            cooc[(p, c)] += self.turn_weight
                            cooc[(c, p)] += self.turn_weight
            prev = current
        return cooc

    def _postprocess(self, emb):
        """平均と上位主成分を引く (どの語にも似てしまう成分) → 単位長に揃える"""
        emb = emb - emb.mean(axis=0, keepdims=True)
        r = min(self.remove_components, emb.shape[1] - 1)
        if r > 0 and emb.shape[0] > 2:
            _, vecs = np.linalg.eigh(emb.T @ emb)  # 昇順なので末尾が最大の主成分
            top = vecs[:, -r:]
            emb = emb - (emb @ top) @ top.T
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        return np.where(norms > 1e-12, emb / np.maximum(norms, 1e-12), 0.0)

    # ------------------------------------------------------------------
    # サブワード補完 (未知語対策)
    # ------------------------------------------------------------------
    def _subwords(self, token: str) -> list[str]:
        """語を構成する文字n-gram。漢字1文字の語はその文字自身。"""
        if len(token) == 1:
            return [token] if "一" <= token <= "鿿" else []
        grams: list[str] = []
        for n in self.sub_ngrams:
            if n <= len(token):
                grams.extend(token[i : i + n] for i in range(len(token) - n + 1))
        return grams

    def _build_subword_table(self, counts) -> None:
        """各n-gramの「そのn-gramを含む語の平均ベクトル」を作る。

        どの語にも入っている汎用n-gram (「して」「ない」など) は df が多いので捨てる。
        残ったn-gramは、珍しいほど (df が小さいほど) 大きな重みで未知語の合成に使う。
        """
        self._ngram_ids, self._ngram_emb, self._ngram_idf = {}, None, None
        containing: dict[str, list[int]] = {}
        for word, i in self.vocab.items():
            for gram in set(self._subwords(word)):
                containing.setdefault(gram, []).append(i)
        size = len(self.words)
        max_df = max(10, int(self.sub_max_df_ratio * size))
        grams: list[str] = []
        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []
        dfs: list[int] = []
        for gram, ids in containing.items():
            if len(ids) > max_df:
                continue
            row = len(grams)
            grams.append(gram)
            dfs.append(len(ids))
            weights = [math.sqrt(counts[i]) for i in ids]
            total = sum(weights)
            for i, w in zip(ids, weights):
                rows.append(row)
                cols.append(i)
                vals.append(w / total)
        if not grams:
            return
        proj = coo_matrix((vals, (rows, cols)), shape=(len(grams), size)).tocsr()
        gram_emb = np.asarray(proj @ self.embeddings)
        norms = np.linalg.norm(gram_emb, axis=1, keepdims=True)
        self._ngram_emb = np.where(norms > 1e-12, gram_emb / np.maximum(norms, 1e-12), 0.0).astype(np.float32)
        self._ngram_idf = np.log(1.0 + size / np.array(dfs, dtype=np.float64)).astype(np.float32)
        self._ngram_ids = {g: i for i, g in enumerate(grams)}

    def _backoff_vector(self, token: str):
        """語彙に無い語を、含まれるn-gramのベクトルから合成する。手がかりが無ければ None。"""
        if self._ngram_emb is None:
            return None
        ids = [self._ngram_ids[g] for g in self._subwords(token) if g in self._ngram_ids]
        if not ids:
            return None
        vec = (self._ngram_idf[ids, None] * self._ngram_emb[ids]).sum(axis=0)
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 1e-12 else None

    # ------------------------------------------------------------------
    # 単語
    # ------------------------------------------------------------------
    def word_vector(self, token: str, *, subword: bool = True):
        """語のベクトル (単位長)。語彙に無ければ、分割した断片の平均 → (subword=True なら) n-gram合成。"""
        if not self.ready:
            return None
        i = self.vocab.get(token)
        if i is not None:
            return self.embeddings[i]
        pieces = [self.vocab[t] for t in self._tokens(token) if t in self.vocab]
        if pieces:
            vec = self.embeddings[pieces].mean(axis=0)
            norm = float(np.linalg.norm(vec))
            if norm > 1e-12:
                return vec / norm
        return self._backoff_vector(token) if subword else None

    def similarity(self, word1: str, word2: str, *, subword: bool = False) -> float:
        """2語のコサイン類似度。既定では未知語のn-gram合成は使わない (置換の誤爆を避ける)。"""
        v1, v2 = self.word_vector(word1, subword=subword), self.word_vector(word2, subword=subword)
        if v1 is None or v2 is None:
            return 0.0
        return float(v1 @ v2)

    calculate_similarity = similarity  # ContextualWordReplacer 向けの互換名

    def _candidate_mask(self, min_count: int, max_doc_ratio: float, min_chars: int):
        mask = self.counts >= min_count
        if self.doc_total:
            mask &= self.doc_freq <= max_doc_ratio * self.doc_total
        if min_chars > 1:
            mask &= self._lengths >= min_chars
        return mask

    def arithmetic(
        self,
        positive: list[str] | None = None,
        negative: list[str] | None = None,
        top_n: int = 5,
        *,
        min_count: int = 2,
        max_doc_ratio: float = 0.2,
        min_chars: int = 2,
        min_sim: float = 0.0,
        **_legacy,
    ) -> list[tuple[str, float]]:
        """単語の演算 (例: 王様 - 男 + 女)。入力語を含む/入力語に含まれる語は除く。

        入力語が語彙に無くても、サブワード補完で近似ベクトルが作れれば計算する。
        """
        positive = list(positive or [])
        negative = list(negative or [])
        if not self.ready or not (positive or negative):
            return []
        inputs = set(positive) | set(negative)
        target = np.zeros(self.embeddings.shape[1])
        for sign, group in ((1.0, positive), (-1.0, negative)):
            for word in group:
                vec = self.word_vector(word, subword=True)
                if vec is None:
                    logger.warning("単語 '%s' はまだ記憶(学習データ)に存在せず、近似もできません。", word)
                    return []
                target += sign * vec
        norm = np.linalg.norm(target)
        if norm < 1e-12:
            return []

        scores = self.embeddings @ (target / norm)
        mask = self._candidate_mask(min_count, max_doc_ratio, min_chars)
        for word in inputs:
            i = self.vocab.get(word)
            if i is not None:
                mask[i] = False
        for i in np.flatnonzero(mask):
            candidate = self.words[i]
            if any(inp in candidate or candidate in inp for inp in inputs):
                mask[i] = False
        n = min(top_n, int(mask.sum()))
        if n <= 0:
            return []
        scores = np.where(mask, scores, -np.inf)
        top = np.argpartition(-scores, n - 1)[:n]
        top = top[np.argsort(-scores[top])]
        return [(self.words[i], float(scores[i])) for i in top if scores[i] > min_sim]

    def most_similar(self, word: str, top_n: int = 5, **kwargs) -> list[tuple[str, float]]:
        return self.arithmetic(positive=[word], top_n=top_n, **kwargs)

    # ------------------------------------------------------------------
    # 文
    # ------------------------------------------------------------------
    def _encode(self, token_lists: list[list[str]], remove_pc: bool = True):
        """語リストの束を文ベクトル (行ごとに単位長、語が無い行は零) にする。

        語ベクトルのSIF重み付き和。未知語はサブワード補完のベクトルを oov_weight 倍して足す。
        """
        n, dim, size = len(token_lists), self.embeddings.shape[1], len(self.words)
        out = np.zeros((n, dim), dtype=np.float32)
        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []
        oov_scale = self.oov_weight * self._oov_sif
        for r, tokens in enumerate(token_lists):
            for token in tokens:
                i = self.vocab.get(token)
                if i is not None:
                    rows.append(r)
                    cols.append(i)
                    vals.append(self._weights[i])
                elif oov_scale > 0:
                    vec = self._backoff_vector(token)
                    if vec is not None:
                        out[r] += oov_scale * vec
        if rows:
            weights = coo_matrix((vals, (rows, cols)), shape=(n, size)).tocsr()
            out += np.asarray(weights @ self.embeddings, dtype=np.float32)

        def unit(mat):
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            return np.where(norms > 1e-6, mat / np.maximum(norms, 1e-6), 0.0).astype(np.float32)

        out = unit(out)
        if remove_pc and self._sent_pc is not None:
            out = unit(out - np.outer(out @ self._sent_pc, self._sent_pc))
        return out

    def _fit_sentence_component(self, sequences: list[list[str]]) -> None:
        """全発言の文ベクトルに共通する第1主成分を求める (話題に関係なく似てしまう成分)"""
        self._sent_pc = None
        dim = self.embeddings.shape[1]
        cov = np.zeros((dim, dim))
        used = 0
        for start in range(0, len(sequences), self._CHUNK):
            chunk = self._encode(sequences[start : start + self._CHUNK], remove_pc=False)
            chunk = chunk[np.linalg.norm(chunk, axis=1) > 0]
            cov += chunk.T.astype(np.float64) @ chunk
            used += len(chunk)
        if used < 3:
            return
        _, vecs = np.linalg.eigh(cov)
        self._sent_pc = vecs[:, -1].astype(np.float32)

    def embed_texts(self, texts: Sequence[str]):
        """発言のリストを (件数, dim) の行列にする。会話ではない行や語の無い発言は零ベクトル。"""
        if not self.ready:
            return None
        parts = []
        for start in range(0, len(texts), self._CHUNK):
            batch = texts[start : start + self._CHUNK]
            token_lists = [self._tokens(t) if self._usable_text(t) else [] for t in batch]
            parts.append(self._encode(token_lists))
        if not parts:
            return np.zeros((0, self.embeddings.shape[1]), dtype=np.float32)
        return parts[0] if len(parts) == 1 else np.vstack(parts)

    def embed_text(self, text: str):
        """1文の意味ベクトル (単位長)。語が1つも拾えなければ None。"""
        if not self.ready:
            return None
        if text in self._sentence_cache:
            return self._sentence_cache[text]
        vec = self.embed_texts([text])[0]
        result = vec if float(np.linalg.norm(vec)) > 0 else None
        if len(self._sentence_cache) >= 50_000:
            self._sentence_cache.clear()
        self._sentence_cache[text] = result
        return result


# ---------------------------------------------------------------------------
# 密ベクトル索引 (記憶の全発言の意味ベクトル)
# ---------------------------------------------------------------------------
class DenseIndex:
    """記憶の各発言を意味ベクトルにして行列で持つ。TfidfIndex と同じく増分更新に対応する。

    - 末尾への追加は新しい分だけ計算して足す (容量は倍々に確保するのでコピーが少ない)
    - 先頭削除・途中挿入は invalidate() で全再構築を予約する
    - 意味空間が学習し直されたら (version が変わったら) 自動で全再構築する
    """

    def __init__(self):
        self._buf = None
        self.size = 0
        self.version = -1
        self._stale = True

    def invalidate(self):
        self._stale = True

    @property
    def matrix(self):
        return None if self._buf is None else self._buf[: self.size]

    def sync(self, entries: Sequence, space: SemanticSpace):
        if np is None or not space.ready:
            self._buf, self.size, self.version, self._stale = None, 0, -1, True
            return
        n = len(entries)
        usable = not self._stale and self.version == space.version and self._buf is not None
        if usable and n == self.size:
            return
        if usable and n > self.size:
            self._append(space.embed_texts([_entry_text(e) for e in entries[self.size :]]))
            return
        self._buf = np.array(space.embed_texts([_entry_text(e) for e in entries]), dtype=np.float32)
        self.size = n
        self.version = space.version
        self._stale = False

    def _append(self, new):
        need = self.size + len(new)
        if need > len(self._buf):
            grown = np.zeros((max(need, int(len(self._buf) * 1.5) + 16), self._buf.shape[1]), dtype=np.float32)
            grown[: self.size] = self._buf[: self.size]
            self._buf = grown
        self._buf[self.size : need] = new
        self.size = need

    def similarities(self, indices: Sequence[int], query) -> list[float]:
        """指定した発言たちと query の意味の近さ (負は0に切る)"""
        matrix = self.matrix
        if matrix is None or query is None or not len(indices):
            return [0.0] * len(indices)
        valid = [i if 0 <= i < self.size else 0 for i in indices]
        sims = matrix[valid] @ query
        return [max(0.0, float(s)) if 0 <= i < self.size else 0.0 for s, i in zip(sims, indices)]

    def search(self, query, top_k: int, min_sim: float) -> list[tuple[float, int]]:
        """query に意味が近い発言を (類似度, idx) の降順で返す"""
        matrix = self.matrix
        if matrix is None or query is None or self.size == 0 or top_k <= 0:
            return []
        sims = matrix @ query
        k = min(top_k, len(sims))
        top = np.argpartition(-sims, k - 1)[:k] if k < len(sims) else np.arange(len(sims))
        top = top[np.argsort(-sims[top])]
        return [(float(sims[i]), int(i)) for i in top if sims[i] >= min_sim]


# ---------------------------------------------------------------------------
# TF-IDF索引
# ---------------------------------------------------------------------------
class TfidfIndex:
    """転置索引つきTF-IDF索引。

    末尾への追加は増分で反映し、IDFのズレを抑えるため一定件数ごとに全再構築する。
    先頭削除や途中挿入が起きたときは invalidate() で全再構築を予約する。
    """

    def __init__(self, rebuild_interval: int = 500):
        self.rebuild_interval = rebuild_interval
        self.doc_count = 0
        self.df: Counter = Counter()
        self.postings: dict[str, list[int]] = {}
        self.vectors: list[dict[str, float]] = []
        self._since_rebuild = 0
        self._stale = True

    def invalidate(self):
        self._stale = True

    def _idf(self, doc_freq: int) -> float:
        return math.log((self.doc_count + 1) / (doc_freq + 1)) + 1.0

    def _vectorize(self, tokens: list[str], idf_cache: dict[str, float] | None = None) -> dict[str, float]:
        vector: dict[str, float] = {}
        for token, count in Counter(tokens).items():
            doc_freq = self.df.get(token)
            if not doc_freq:
                continue  # 索引に存在しないトークンは無視
            idf = idf_cache[token] if idf_cache is not None else self._idf(doc_freq)
            vector[token] = count * idf
        return _normalize_vector(vector)

    def rebuild(self, entries: Sequence):
        df: Counter = Counter()
        postings: dict[str, list[int]] = {}
        doc_tokens: list[list[str]] = []

        for idx, entry in enumerate(entries):
            tokens = tokenize_ngrams(_entry_text(entry))
            doc_tokens.append(tokens)
            unique = set(tokens)
            df.update(unique)
            for token in unique:
                postings.setdefault(token, []).append(idx)

        self.doc_count = len(doc_tokens)
        self.df = df
        self.postings = postings
        idf_cache = {token: self._idf(freq) for token, freq in df.items()}
        self.vectors = [self._vectorize(tokens, idf_cache) for tokens in doc_tokens]
        self._since_rebuild = 0
        self._stale = False

    def _append(self, entries: Sequence):
        for entry in entries:
            tokens = tokenize_ngrams(_entry_text(entry))
            idx = self.doc_count
            self.doc_count += 1
            unique = set(tokens)
            self.df.update(unique)
            for token in unique:
                self.postings.setdefault(token, []).append(idx)
            self.vectors.append(self._vectorize(tokens))
        self._since_rebuild += len(entries)

    def sync(self, entries: Sequence):
        """記憶の現状に索引を合わせる。変化がなければ何もしない。"""
        size = len(entries)
        if not self._stale and size == self.doc_count:
            return
        added = size - self.doc_count
        if not self._stale and 0 < added and self._since_rebuild + added <= self.rebuild_interval:
            self._append(entries[self.doc_count :])
        else:
            self.rebuild(entries)

    def query_vector(self, tokens: list[str]) -> dict[str, float]:
        return self._vectorize(tokens)

    def cosine(self, query_vector: dict[str, float], idx: int) -> float:
        if not query_vector or not 0 <= idx < len(self.vectors):
            return 0.0
        return _sparse_dot(query_vector, self.vectors[idx])

    def match_counts(self, query_tokens: Iterable[str]) -> Counter:
        """各文書がクエリのトークンを何種類含むか {idx: 一致数}"""
        counts: Counter = Counter()
        for token in set(query_tokens):
            counts.update(self.postings.get(token, ()))
        return counts


@dataclass(slots=True)
class _Candidate:
    idx: int
    text: float
    good: bool
    context: float = 0.0
    history: float = 0.0
    style: float = 0.0
    semantic: float = 0.0
    final: float = 0.0


# ---------------------------------------------------------------------------
# 単語置き換えエンジン
# ---------------------------------------------------------------------------
class ContextualWordReplacer:
    """文脈を考慮したトークンベースの単語置換エンジン"""

    # 置換ルールとして採用する大きさの範囲。
    # 文全体が丸ごと入れ替わるような差分や、1文字だけの差分 (「は」→「が」など)
    # をルールにすると、無関係な箇所まで書き換えてしまうので捨てる。
    MIN_RULE_CHARS = 2
    MAX_RULE_CHARS = 16
    MAX_RULE_TOKENS = 4

    def __init__(self, semantic_learner, char_sim=0.85, sem_sim=0.75, protected_words=None, commonness_func=None):
        self.semantic_learner = semantic_learner
        self.char_sim = char_sim
        self.sem_sim = sem_sim
        # 名前など。これらは完全一致のときしか置換しない (「花子」と「次郎」は
        # 文脈が似ているので、意味ベクトルの類似度で曖昧一致させると取り違える)
        self.protected_words = protected_words if protected_words is not None else set()
        # 文字列 -> 「その文字列を含む発言の割合」(記憶内の出現頻度)。None なら頻度による判断はしない
        self.commonness_func = commonness_func
        self._rule_cache: dict[tuple, list[dict]] = {}

    MIN_ANCHOR_CHARS = 2  # 差分の前後で、これ以上続く共通部分だけを「変わっていない部分」とみなす
    # 差分と差分の「間」にある共通部分は、これ以上長いときだけ「変わっていない部分」とみなす。
    # 短い共通部分は偶然の一致のことが多い (Pyth-on と Comm-on の "on" など)。そこで語を切ると
    # 「Pyth→Comm」のような切れ端のルールになるので、間の共通部分ごとひとまとまりの置換にする
    # (ルールは長くなるが、より限定的になるだけで壊れない)
    MIN_INTERIOR_ANCHOR_CHARS = 3
    RULE_CACHE_SIZE = 4096
    # 語の途中から強制的に切り出したトークンで置換するときの安全策。
    # 記憶の中でありふれていて、しかも短い文字列 (「った」「ない」など、文法の断片) は、他の語の一部に
    # 偶然入っているだけのことが多いので、周辺トークンも2つ以上一致しているときだけ置換する。
    # それ以外 (「Python」「C++」など、長い内容語) は、周辺が違っていても置換してよい
    # (入力の語を返信に差し替える、という本来の使い方では、周辺は必ず違うため。話題の中心で頻出でも止めない)
    FORCED_SPLIT_MIN_CONTEXT = 2
    FORCED_SPLIT_COMMON_RATIO = 0.05  # この割合以上の発言に出る文字列を「ありふれている」とみなす
    FORCED_SPLIT_SHORT_CHARS = 3  # この文字数以下を「短い」とみなす

    def _is_common(self, text: str) -> bool:
        """記憶の中でありふれていて短い文字列か。頻度が分からないとき (commonness_func が無い) は False"""
        if self.commonness_func is None or len(text) > self.FORCED_SPLIT_SHORT_CHARS:
            return False
        return self.commonness_func(text) >= self.FORCED_SPLIT_COMMON_RATIO

    @staticmethod
    def _is_ideograph(text: str) -> bool:
        """漢字1文字 (「猫」「車」など。1文字でも単語として置換してよい)"""
        return len(text) == 1 and "一" <= text <= "鿿"

    def _is_usable_rule(self, old_tokens: list[str], new_tokens: list[str]) -> bool:
        old_text, new_text = "".join(old_tokens), "".join(new_tokens)
        if not new_text or old_text == new_text:
            return False
        # 1文字トークンが2つ以上並んだだけの断片は、トークナイザが「まとまり」と認めていない文字列の切れ端。
        # (チャンネルIDの数字 "5","6" → "0","8" のように、桁の差が語の置換ルールになってしまう)
        if len(old_tokens) >= 2 and all(len(token) == 1 for token in old_tokens):
            return False
        if len(old_text) < self.MIN_RULE_CHARS and not self._is_ideograph(old_text):
            return False
        if len(old_tokens) > self.MAX_RULE_TOKENS or len(new_tokens) > self.MAX_RULE_TOKENS:
            return False
        return max(len(old_text), len(new_text)) <= self.MAX_RULE_CHARS

    def _core_pairs(self, old: str, new: str) -> list[tuple[str, str, bool]]:
        """入れ替わった部分だけを [(旧, 新, 全体が入れ替わったか)] で取り出す。

        トークンの切れ目が文ごとにずれると、「猫ばかり」→「水泳ばかり」や
        「寿司が好きですよ」→「茶道が好きです」のように、助詞や文末ごと
        入れ替わった形で差分が出る。文字単位で比べ、2文字以上続く共通部分を
        「変わっていない部分」として除くと、「猫」→「水泳」「寿司」→「茶道」が取れる。
        1文字だけの共通部分 (「電車」→「電話」の「電」など) は語の一部なので除かない。
        """
        anchor = self.MIN_ANCHOR_CHARS
        spans: list[list[int]] = []  # [旧の開始, 旧の終了, 新の開始, 新の終了]
        for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, old, new, autojunk=False).get_opcodes():
            if tag == "equal":
                continue
            if spans and i1 - spans[-1][1] < self.MIN_INTERIOR_ANCHOR_CHARS:  # 間の共通部分が短い → 同じ差分として繋ぐ
                spans[-1][1], spans[-1][3] = i2, j2
            else:
                spans.append([i1, i2, j1, j2])
        if not spans:
            return []

        first, last = spans[0], spans[-1]
        if 0 < first[0] < anchor and first[0] == first[2]:  # 先頭の短い共通部分も取り込む
            first[0] = first[2] = 0
        tail_old, tail_new = len(old) - last[1], len(new) - last[3]
        if 0 < tail_old < anchor and tail_old == tail_new:  # 末尾の短い共通部分も取り込む
            last[1], last[3] = len(old), len(new)

        return [
            (old[i1:i2], new[j1:j2], i1 == 0 and i2 == len(old) and j1 == 0 and j2 == len(new))
            for i1, i2, j1, j2 in spans
        ]

    def extract_token_replacements(self, before_tokens: list[str], after_tokens: list[str]) -> list[dict]:
        """置換ルールを抽出する。履歴はほぼ毎回同じ内容なので結果を使い回す。"""
        key = (tuple(before_tokens), tuple(after_tokens), len(self.protected_words))  # 名前が増えたら作り直す
        rules = self._rule_cache.get(key)
        if rules is None:
            if len(self._rule_cache) >= self.RULE_CACHE_SIZE:
                self._rule_cache.clear()
            rules = self._rule_cache[key] = self._extract_token_replacements(before_tokens, after_tokens)
        return rules

    def _extract_token_replacements(self, before_tokens: list[str], after_tokens: list[str]) -> list[dict]:
        """トークン単位で置換ペアと、その左右の広い文脈（計12トークン）を抽出する"""
        replacements = []
        matcher = difflib.SequenceMatcher(None, before_tokens, after_tokens)
        
        # 24トークンの文脈を見るため、左右に12トークンずつ確保
        WINDOW_SIZE = 12
        
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'replace':
                old_tokens = before_tokens[i1:i2]
                new_tokens = after_tokens[j1:j2]

                # 1文字トークンが並んだだけの断片 (「C++」→ C,+,+ / 「CommonLisp」→ C,o,m,m,on,L,i,s,p) は、
                # 差が出た部分だけ切り出すと「++」→「ommonLisp」のような語として成り立たない切れ端になる。
                # 隣り合う同じ1文字トークンまで含めて、ひとかたまりの語として扱う。
                # (チャンネルIDのような長い数字列は、全体が長さ制限を超えるので自然に外れる)
                fragment_run = len(old_tokens) >= 2 and all(len(tok) == 1 for tok in old_tokens)
                if fragment_run:
                    while i1 > 0 and j1 > 0 and len(before_tokens[i1 - 1]) == 1 and before_tokens[i1 - 1] == after_tokens[j1 - 1]:
                        i1 -= 1
                        j1 -= 1
                    while i2 < len(before_tokens) and j2 < len(after_tokens) and len(before_tokens[i2]) == 1 and before_tokens[i2] == after_tokens[j2]:
                        i2 += 1
                        j2 += 1
                    old_tokens = ["".join(before_tokens[i1:i2])]
                    new_tokens = ["".join(after_tokens[j1:j2])]

                # 左右6トークンずつの環境（文脈）を保存（インデックスの範囲外はBOS/EOSでパディング）
                left_context = before_tokens[max(0, i1 - WINDOW_SIZE):i1]
                if i1 - WINDOW_SIZE < 0:
                    left_context = ["<BOS>"] * (WINDOW_SIZE - i1) + left_context

                right_context = before_tokens[i2:min(len(before_tokens), i2 + WINDOW_SIZE)]
                if i2 + WINDOW_SIZE > len(before_tokens):
                    right_context = right_context + ["<EOS>"] * (i2 + WINDOW_SIZE - len(before_tokens))

                core_pairs = (
                    [(old_tokens[0], new_tokens[0], True)]  # 断片の並びは、すでに語の単位にしてある
                    if fragment_run
                    else self._core_pairs("".join(old_tokens), "".join(new_tokens))
                )
                for old_core, new_core, whole in core_pairs:
                    # 全体が入れ替わっているなら元のトークン列のまま、一部なら芯だけを1語として扱う
                    # 名前 (登録語) の入れ替えは、文字単位の差分で切らず必ず丸ごと1語として扱う
                    # (「小関琉聖だった人」→「小関琉聖」が「だった人」の削除になって捨てられるのを防ぐ)
                    named = "".join(old_tokens) in self.protected_words or "".join(new_tokens) in self.protected_words
                    rule_old, rule_new = (old_tokens, new_tokens) if (whole or named) else ([old_core], [new_core])
                    # 置換先は出力の連結にしか使わない (マッチングに使うのは置換元だけ) ので、1つの文字列にまとめる。
                    # トークナイザが未知語を1文字ずつに分けても (「CommonLisp」→ 9トークン)、トークン数の上限で捨てない
                    rule_new = ["".join(rule_new)]
                    if not self._is_usable_rule(rule_old, rule_new):
                        continue
                    replacements.append({
                        "old": rule_old,
                        "new": rule_new,
                        "left": left_context,   # 12トークンのリスト
                        "right": right_context, # 12トークンのリスト
                        # 発言まるごとの入れ替えか。一部を取り出しただけのルールは、文脈が全部パディングでも
                        # 「文全体」ではない (「Pythonってどうよ」が1トークンだと、「Python」の前後の文脈が消える)
                        "whole_sentence": bool(whole or named) and i1 == 0 and i2 == len(before_tokens),
                    })
        return replacements

    def _match_context_and_semantics(
        self, tokens: list[str], idx: int, rep: dict, natural: set[str] | None = None
    ) -> bool:
        """単語の類似度だけでなく、『前後12トークンの広域文脈』が一致するかを検証する"""
        old_tokens = rep["old"]
        if idx + len(old_tokens) > len(tokens):
            return False
            
        WINDOW_SIZE = 12
            
        # 1. 対象の単語自体が一致、または類似しているかチェック
        is_exact_match = (tokens[idx : idx + len(old_tokens)] == old_tokens)
        if not is_exact_match:
            for offset, old_w in enumerate(old_tokens):
                target_w = tokens[idx + offset]
                if old_w in self.protected_words or target_w in self.protected_words:
                    return False
                if Levenshtein.normalized_similarity(target_w, old_w) >= self.char_sim:
                    continue
                if self.semantic_learner.calculate_similarity(target_w, old_w) >= self.sem_sim:
                    continue
                return False

        # 2. 【24トークン広域文脈判定】
        # 実際の現在のトークン列から前後12トークンを切り出し（パディング付き）
        actual_left = tokens[max(0, idx - WINDOW_SIZE):idx]
        if idx - WINDOW_SIZE < 0:
            actual_left = ["<BOS>"] * (WINDOW_SIZE - idx) + actual_left
            
        right_start = idx + len(old_tokens)
        actual_right = tokens[right_start : min(len(tokens), right_start + WINDOW_SIZE)]
        if right_start + WINDOW_SIZE > len(tokens):
            actual_right = actual_right + ["<EOS>"] * (right_start + WINDOW_SIZE - len(tokens))
            
        # パディング (<BOS>/<EOS>) 同士の一致は「文脈が一致した」証拠にならないので数えない
        pads = ("<BOS>", "<EOS>")
        actual_has_context = any(tok not in pads for tok in actual_left + actual_right)
        # 文全体が入れ替わった履歴から作ったルール (文脈が全部パディング) は、文全体にだけ適用する
        # (でないと「った→ぃ」のようなルールが、あらゆる文の一部を壊す)
        # ただし話者名・自分の名前 (登録語) は、どの文の中でも置換する
        is_name_rule = any(tok in self.protected_words for tok in old_tokens)
        if rep.get("whole_sentence") and actual_has_context and not is_name_rule:
            return False

        left_matches = sum(1 for a, b in zip(actual_left, rep["left"]) if a == b and a not in pads)
        right_matches = sum(1 for a, b in zip(actual_right, rep["right"]) if a == b and a not in pads)
        context_matches = left_matches + right_matches

        # 完全一致でも、extra_words で語の途中から強制的に切り出したトークンは偶然の一致かもしれない
        # (「まったく」の「った」など)。その場合は、周りの文脈も一部一致していることを求める。
        # 話者名などの登録語は自然に1トークンになるので、この条件の対象にならない。
        forced_split = (
            is_exact_match
            and natural is not None
            and any(tok not in natural for tok in tokens[idx : idx + len(old_tokens)])
        )
        if forced_split and context_matches < self.FORCED_SPLIT_MIN_CONTEXT and self._is_common("".join(old_tokens)):
            return False

        # 通常の単語 (あいまい一致) の場合：
        # 24トークン中、左右合わせて最低でも6トークン以上の文脈一致を求める（環境の安全性を担保）
        if context_matches < 6 and not is_exact_match:
            return False

        return True

    def execute_replace(
        self,
        text: str,
        after_list: list[str],
        before_list: list[str],
        tokenizer_func,
        *,
        self_names: Iterable[str] = (),
        base_speaker: str | None = None,
    ) -> str:
        """文脈付き置換を実行する

        self_names / base_speaker を渡すと、「誰かの名前 → 自分の名前」のルールは
        その人自身の発言 (base_speaker) にだけ適用する。これは「自分の発言の中の自分の名前」を
        直すためのルールなので、別の人の発言の中でその名前が出てきても (呼びかけ・他人の話) 自分の名前には直さない。
        base_speaker が None なら制限しない。
        """
        # すべての置換履歴からトークンベースのルールを構築
        all_rules = []
        for after, before in zip(after_list, before_list):
            if not after or not before:
                continue
            b_tokens = tokenizer_func(before, drop_threshold=0.4)
            a_tokens = tokenizer_func(after, drop_threshold=0.4)
            all_rules.extend(self.extract_token_replacements(b_tokens, a_tokens))
            
        if base_speaker is not None:
            mine = set(self_names)
            all_rules = [
                r for r in all_rules
                if not (
                    "".join(r["new"]) in mine
                    and "".join(r["old"]) in self.protected_words
                    and "".join(r["old"]) != base_speaker
                )
            ]

        # 最新の履歴を優先
        all_rules.reverse()
        
        # 入力文をトークナイズ。置換したい語は、助詞などとくっついていても
        # 単独のトークンとして切り出す (でないとルールと照合できない)
        targets = {rule["old"][0] for rule in all_rules if len(rule["old"]) == 1}
        tokens = tokenizer_func(text, drop_threshold=0.4, extra_words=targets)
        natural = set(tokenizer_func(text, drop_threshold=0.4)) if targets else None  # 強制切り出し無しの自然な分割
        i = 0
        while i < len(tokens):
            replaced = False
            for rule in all_rules:
                if self._match_context_and_semantics(tokens, i, rule, natural):
                    logger.info(f"文脈置換適用: {tokens[i:i+len(rule['old'])]} -> {rule['new']} (文脈: {rule['left']}_{rule['right']})")
                    # トークンリストを書き換え
                    tokens[i:i+len(rule['old'])] = rule['new']
                    i += len(rule['new'])
                    replaced = True
                    break
            if not replaced:
                i += 1
                
        return "".join(tokens)

# ---------------------------------------------------------------------------
# 本体
# ---------------------------------------------------------------------------
class Himawaria:
    """会話記憶の検索・学習・応答を管理するクラス。"""

    def __init__(
        self,
        directory: str,
        maximum_word_replacer_memory: int = 128,
        min_similarity_threshold: float = 0.45,  # 互換性のため保持 (現在は未使用)
        *,
        config: SearchConfig | None = None,
        max_sentences: int = MAX_SENTENCES,
        autosave_every: int = 1,
        semantic_rebuild_interval: int = 50,
        rng: random.Random | None = None,
    ):
        self.direc = Path(directory)
        self.direc.mkdir(parents=True, exist_ok=True)

        self.maximum_word_replacer_memory = maximum_word_replacer_memory
        self.min_similarity_threshold = min_similarity_threshold
        self.config = config or SearchConfig()
        self.max_sentences = max_sentences
        self.autosave_every = max(1, autosave_every)
        self.semantic_rebuild_interval = semantic_rebuild_interval
        self._rng = rng or random.Random()

        self.memory: dict = {}
        self.settings: dict = {}
        self.chara_style = ""

        self.last_bot_response = ""
        self.last_bot_base = ""
        self.last_input_content = ""
        self.last_input_similar = ""
        self.last_bot_base_user = ""
        self.last_input_user = ""
        self.last_input_similar_user = ""
        self.pre_heart = 0
        self.heart = 0
        self.last_user = "あんた"
        self.last_user_replied = "あんた"
        self.current_voice: str | None = None
        self.user_log: list = [None] * 10
        self.word_replacer_memory_after: list[str] = []
        self.word_replacer_memory_before: list[str] = []
        self.last_previous_input_user = ""
        self.last_previous_input_content = ""
        self.rate = 1.0
        self.latest_log: list = [["init", "init"]]

        self.tokenizer = NgramTokenizer()
        self.semantic_learner = SemanticSpace()  # 名前は互換のため据え置き
        self.index = TfidfIndex()
        self.dense = DenseIndex()
        self.contextual_word_replacer = ContextualWordReplacer(
            semantic_learner=self.semantic_learner,
            protected_words=self.tokenizer.forced_words,
            commonness_func=self._string_doc_ratio,
        )
        self._commonness_cache: dict[str, float] = {}
        self._commonness_stamp = 0
        self._semantic_trained_size: int | None = None
        self._unsaved = 0
        self._main_is_valid = False

        self._load_initial_data()
        self._init_tokenizer()
        # 話者名・自分の名前は本文中でも必ず1語として切り出す (名前置換の前提)
        self._register_names(entry[1] for entry in self.memory["sentence"] if len(entry) >= 2)

        sentences = self.memory["sentence"]
        if sentences:
            self.heart = self._rng.randint(0, len(sentences) - 1)

    # ------------------------------------------------------------------
    # 読み込み・保存
    # ------------------------------------------------------------------
    @property
    def my_names(self) -> list[str]:
        return [n for n in self.settings.get("mynames", "").split("|") if n]

    def _load_initial_data(self):
        """設定・記憶・キャラクター口調ファイルの読み込み"""
        memory_path = self.direc / "memory.json"
        backup_path = self.direc / "memory_backup.json"
        settings_path = self.direc / "settings.json"
        chara_path = self.direc / "chara_style.txt"

        if settings_path.exists():
            with open(settings_path, "r", encoding="utf8") as f:
                self.settings = json.load(f)
        else:
            self.settings = {"mynames": "ひまわり|ヒマワリ"}

        self.memory = {}
        for path in (memory_path, backup_path):
            if not path.exists():
                continue
            try:
                with open(path, "r", encoding="utf8") as f:
                    data = json.load(f)
            except (OSError, ValueError) as e:
                logger.warning("記憶ファイルの読み込みに失敗: %s (%s)", path, e)
                continue
            if isinstance(data, dict) and "sentence" in data:
                self.memory = data
                self._main_is_valid = path == memory_path
                break
            logger.warning("記憶ファイルの形式が不正です: %s", path)

        self.memory.setdefault("sentence", [])
        self.memory.setdefault("word_replacer_memory_after", [])
        self.memory.setdefault("word_replacer_memory_before", [])
        self.word_replacer_memory_after = self.memory["word_replacer_memory_after"]
        self.word_replacer_memory_before = self.memory["word_replacer_memory_before"]

        if chara_path.exists():
            self.chara_style = chara_path.read_text(encoding="utf8")
        else:
            self.chara_style = ""
            chara_path.write_text("", encoding="utf8")

    def _init_tokenizer(self):
        """保存済みモデルがあれば読み込み、なければ記憶から学習する"""
        candidates = (
            (self.direc / TOKENIZER_FILE, self.tokenizer.load),
            (self.direc / LEGACY_TOKENIZER_FILE, self.tokenizer.load_legacy_pickle),
        )
        for path, loader in candidates:
            if not path.exists():
                continue
            try:
                loader(path)
                return
            except Exception as e:  # 壊れたモデルは捨てて学習し直す
                logger.warning("トークナイザーモデルの読み込みに失敗しました: %s (%s)", path, e)
        self._train_tokenizer_from_memory()

    def _train_tokenizer_from_memory(self):
        self.tokenizer.train(
            entry[0]
            for entry in self.memory["sentence"]
            if len(entry) >= 2 and _is_trainable(entry[0], entry[1])
        )

    @staticmethod
    def _write_json_atomic(path: Path, data: dict, backup_path: Path | None = None):
        """一時ファイルに書いてから置き換える。旧ファイルはバックアップへ退避する。"""
        tmp_path = path.with_name(path.name + ".tmp")
        with open(tmp_path, "w", encoding="utf8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4, sort_keys=True, separators=(',', ': '))
        if backup_path is not None and path.exists():
            os.replace(path, backup_path)
        os.replace(tmp_path, path)

    def save_data(self):
        """記憶とトークナイザーを保存する"""
        memory_path = self.direc / "memory.json"
        backup_path = self.direc / "memory_backup.json"
        # 本体が壊れている可能性がある間は、正常なバックアップを上書きしない
        self._write_json_atomic(
            memory_path, self.memory, backup_path if self._main_is_valid else None
        )
        self._main_is_valid = True
        self.tokenizer.save(self.direc / TOKENIZER_FILE)
        self._unsaved = 0

    # ------------------------------------------------------------------
    # 学習
    # ------------------------------------------------------------------
    def learn_sentence(self, x: str, u: str, save: bool = True, direct_learning: bool = False):
        """会話を記憶に追加する"""
        if u in self.my_names:
            speaker = "!output"
        elif direct_learning and u not in RESERVED_SPEAKERS:
            speaker = "!input-" + u
        else:
            speaker = u
        self._register_names([speaker])

        sentences = self.memory["sentence"]
        sentences.append([x, speaker])
        if _is_trainable(x, speaker):
            self.tokenizer.train([x])

        overflow = len(sentences) - self.max_sentences + 1
        if overflow > 0:
            del sentences[:overflow]
            self.heart = max(0, self.heart - overflow)
            self.pre_heart = max(0, self.pre_heart - overflow)
            self._invalidate_indexes()

        self._unsaved += 1
        if save and self._unsaved >= self.autosave_every:
            self.save_data()

    def record(self):
        if self.current_voice:
            self.learn_sentence(self.current_voice, "!")
            self.evaluate()

    def evaluate(self):
        """フィードバック(!good / !bad)の評価"""
        sentences = self.memory["sentence"]
        if not sentences or sentences[-1][0] in SYSTEM_MARKERS:
            return
        if self.heart + 1 < len(sentences) - 1:
            flag = sentences[self.heart + 1][0]
            if flag in SYSTEM_MARKERS:
                logger.info("このメッセージは%s", "良い" if flag == "!good" else "悪い")
                self.learn_sentence(flag, "!system")

    # ------------------------------------------------------------------
    # 索引
    # ------------------------------------------------------------------
    def update_tfidf_index(self):
        """TF-IDF索引と意味ベクトルを強制的に全再構築する"""
        self.index.rebuild(self.memory["sentence"])
        self._ensure_semantic(force=True)
        self.dense.sync(self.memory["sentence"], self.semantic_learner)

    COMMONNESS_MIN_SENTENCES = 200  # 記憶がこれより少ないと頻度は当てにならないので、ありふれているとは判断しない

    def _string_doc_ratio(self, text: str) -> float:
        """記憶の発言のうち、text を含むものの割合。結果は記憶が1割増えるまで使い回す。"""
        sentences = self.memory["sentence"]
        size = len(sentences)
        if size < self.COMMONNESS_MIN_SENTENCES:
            return 0.0
        if abs(size - self._commonness_stamp) > 0.1 * max(size, 1):
            self._commonness_cache.clear()
            self._commonness_stamp = size
        ratio = self._commonness_cache.get(text)
        if ratio is None:
            ratio = sum(1 for entry in sentences if text in _entry_text(entry)) / size
            self._commonness_cache[text] = ratio
        return ratio

    def _invalidate_indexes(self):
        """先頭削除・途中挿入でidxがずれたとき、両方の索引の全再構築を予約する"""
        self.index.invalidate()
        self.dense.invalidate()

    def _ensure_index(self):
        self.index.sync(self.memory["sentence"])
        self._ensure_semantic()
        self.dense.sync(self.memory["sentence"], self.semantic_learner)

    def _ensure_semantic(self, force: bool = False):
        """意味ベクトルは重いので、記憶が一定量増減したときだけ再学習する"""
        sentences = self.memory["sentence"]
        trained = self._semantic_trained_size
        if not force and trained is not None:
            threshold = max(self.semantic_rebuild_interval, trained // 20)
            if abs(len(sentences) - trained) < threshold:
                return
        self.semantic_learner.train_from_memory(sentences, self.tokenizer)
        self._semantic_trained_size = len(sentences)

    # ------------------------------------------------------------------
    # 返信探索
    # ------------------------------------------------------------------
    def is_next_ok(self) -> bool:
        sentences = self.memory["sentence"]
        if len(sentences) - 1 <= self.heart + 1:
            return False
        next_sen = sentences[self.heart + 1]
        curr_sen = sentences[self.heart]
        return (
            self.last_input_content != next_sen[0]
            and self.last_bot_response != next_sen[0]
            and next_sen[1] == curr_sen[1]
            and next_sen[1] != "!"
            and "!system" not in next_sen[1]
        )

    def is_available(self, d: float, b: int, type: int = 1) -> bool:  # noqa: A002 (互換性のため)
        sentences = self.memory["sentence"]
        tag = sentences[b + 2][0] if b + 2 < len(sentences) - 1 else None
        if type == 0:
            return tag == "!good" and d >= 0.6
        if type == 1:
            return tag != "!bad"
        return False

    def _has_good_tag(self, idx: int) -> bool:
        sentences = self.memory["sentence"]
        return any(
            idx + offset < len(sentences) and sentences[idx + offset][0] == "!good"
            for offset in (1, 2)
        )

    @staticmethod
    def _is_duplicate(reply: str, last_input: str, last_bot_resp: str, last_bot_base: str) -> bool:
        if len(reply) <= 2:
            return reply in (last_bot_resp, last_bot_base)
        return any(
            Levenshtein.normalized_similarity(reply, target) >= DUPLICATE_SIMILARITY
            for target in (last_input, last_bot_resp, last_bot_base)
        )

    def _is_valid_pair(self, idx: int) -> bool:
        """sentences[idx] -> sentences[idx+1] が返信ペアとして使えるか"""
        sentences = self.memory["sentence"]
        if idx < 0 or idx + 1 >= len(sentences):
            return False
        source, response = sentences[idx], sentences[idx + 1]
        response_text = response[0]
        if _is_system_or_marker(response):
            return False
        if response_text is None or not str(response_text).strip():
            return False
        if str(response[1]) == str(source[1]):
            return False
        return not self._is_duplicate(
            str(response_text),
            str(self.last_input_content or ""),
            str(self.last_bot_response or ""),
            str(self.last_bot_base or ""),
        )

    def _collect_candidates(
        self, query_tokens: list[str], query_semantic=None
    ) -> list[tuple[float, int]]:
        """(TF-IDFスコア, idx) の候補を返す。文字の一致 (TF-IDF) と意味の近さ (密ベクトル) の2経路。

        1. TF-IDF : 転置索引で絞り込み、コサイン類似度で採点する
        2. 意味検索: 入力文の意味ベクトルに近い発言を行列積で探す。文字がほとんど一致しなくても拾える。
                     TF-IDF側で拾えなかった分だけを少数追加する。

        返す値は常に「TF-IDFのコサイン」で、意味側の値は混ぜない (尺度が違うので、混ぜると後段の
        正規化や min_text_score の意味が崩れる)。意味の近さは _score_candidates が別の欄で持つ。
        意味だけで拾った候補の TF-IDF スコアは、実際に計算した値 (文字が合わなければ0に近い)。
        """
        cfg = self.config
        candidates = self._lexical_candidates(query_tokens)

        if query_semantic is not None and cfg.dense_candidates > 0:
            seen = {idx for _, idx in candidates}
            query_vector = self.index.query_vector(query_tokens)
            added = 0
            for _sim, idx in self.dense.search(query_semantic, cfg.dense_candidates * 4, cfg.dense_min_sim):
                if idx in seen or not self._is_valid_pair(idx):
                    continue
                text_score = self.index.cosine(query_vector, idx)
                if self._has_good_tag(idx):
                    text_score *= cfg.good_tag_multiplier  # 文字一致の候補と同じ扱い
                candidates.append((text_score, idx))
                added += 1
                if added >= cfg.dense_candidates:
                    break
        return candidates

    def _lexical_candidates(self, query_tokens: list[str]) -> list[tuple[float, int]]:
        """転置索引で絞り込み、(TF-IDFスコア, idx) の候補を返す。"""
        cfg = self.config
        query_set = set(query_tokens)
        query_vector = self.index.query_vector(query_tokens)
        if not query_vector:
            return []

        min_matches = min(cfg.min_query_matches, len(query_set))
        check_coverage = len(query_set) >= 4

        # 1) 安いチェック(一致数・コサイン)だけで全候補をふるいにかける
        scored: list[tuple[float, int]] = []
        for idx, match_count in self.index.match_counts(query_set).items():
            if match_count < min_matches:
                continue
            if check_coverage and match_count / len(query_set) < cfg.min_coverage:
                continue
            score = self.index.cosine(query_vector, idx)
            if score <= cfg.min_tfidf_candidate_score:
                continue
            if self._has_good_tag(idx):
                score *= cfg.good_tag_multiplier
            scored.append((score, idx))

        # 2) 高スコア順に、重い妥当性チェック(編集距離)を通った上位だけを残す
        scored.sort(reverse=True)
        limit = cfg.max_scored_candidates
        candidates: list[tuple[float, int]] = []
        for score, idx in scored:
            if self._is_valid_pair(idx):
                candidates.append((score, idx))
                if 0 < limit <= len(candidates):
                    break
        return candidates

    def _recent_user_utterances(self, x: str, u: str) -> list[str]:
        """直近のユーザー発言 (新しい順、今回の入力は除く)"""
        log = self.latest_log
        prior = log[:-1] if log and log[-1] == [x, u] else log
        limit = len(self.config.history_decay_weights)

        history: list[str] = []
        for item in reversed(prior):
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            text = item[0]
            if not isinstance(text, str) or not text.strip() or not _is_user_entry(item):
                continue
            history.append(text)
            if len(history) >= limit:
                break
        return history

    def _context_score(self, idx: int) -> float:
        tokens: list[str] = []
        sentences = self.memory["sentence"]
        for j in range(max(0, idx - 4), idx):
            if _is_user_entry(sentences[j]):
                tokens.extend(tokenize_ngrams(_entry_text(sentences[j])))
        return self.index.cosine(self.index.query_vector(tokens), idx)

    def _history_score(self, idx: int, history_vectors: list[dict[str, float]]) -> float:
        """候補の直前のユーザー発言と、実際の直近会話の近さ (古いほど減衰)"""
        if not history_vectors:
            return 0.0
        cfg = self.config
        sentences = self.memory["sentence"]
        score = 0.0
        k = 0
        pos = idx - 1
        floor = max(0, idx - cfg.history_lookback)
        while pos >= floor and k < len(history_vectors):
            if _is_user_entry(sentences[pos]):
                score += self.index.cosine(history_vectors[k], pos) * cfg.history_decay_weights[k]
                k += 1
            pos -= 1
        return score

    def _score_candidates(
        self, candidates: list[tuple[float, int]], x: str, u: str, query_semantic=None
    ) -> list[_Candidate]:
        cfg = self.config

        history_vectors: list[dict[str, float]] = []
        if cfg.history_score_weight:
            history_vectors = [
                self.index.query_vector(tokenize_ngrams(text))
                for text in self._recent_user_utterances(x, u)
            ]
        style_vector: dict[str, float] = {}
        if cfg.chara_style_weight and self.chara_style:
            style_vector = self.index.query_vector(tokenize_ngrams(self.chara_style))
        # 候補すべての「入力との意味の近さ」を、密索引の行列積でまとめて求める
        semantics = (
            self.dense.similarities([idx for _, idx in candidates], query_semantic)
            if query_semantic is not None
            else [0.0] * len(candidates)
        )
        floor = min(max(cfg.semantic_score_floor, 0.0), 0.99)

        items = [
            _Candidate(
                idx=idx,
                text=text_score,
                good=self._has_good_tag(idx),
                context=self._context_score(idx) if cfg.context_score_weight else 0.0,
                history=self._history_score(idx, history_vectors),
                style=self.index.cosine(style_vector, idx + 1) if style_vector else 0.0,
                semantic=semantics[n],
            )
            for n, (text_score, idx) in enumerate(candidates)
        ]

        text_norm = _normalize_scores([it.text for it in items])
        context_norm = _normalize_scores([it.context for it in items])
        history_norm = _normalize_scores([it.history for it in items])
        style_norm = _normalize_scores([it.style for it in items])

        for i, it in enumerate(items):
            it.final = (
                cfg.text_score_weight * text_norm[i]
                + cfg.context_score_weight * context_norm[i]
                + cfg.history_score_weight * history_norm[i]
                + (cfg.good_tag_bonus if it.good else 0.0)
                + cfg.chara_style_weight * style_norm[i]
                # 意味は候補間の相対 (min-max) ではなく絶対尺度で採点する。全員が意味的に遠ければ加点も0のまま
                + cfg.semantic_score_weight * max(0.0, (it.semantic - floor) / (1.0 - floor))
            )
        return items

    def _pick_best(self, scored: list[_Candidate], force: bool) -> _Candidate | None:
        cfg = self.config
        top = heapq.nlargest(max(2, cfg.near_tie_pool_size), scored, key=lambda it: it.final)
        best = top[0]

        # 返信の根拠は「文字が合っている」か「意味が十分近い」のどちらか。尺度の違う2つの値は混ぜず、別々に判定する
        lexical_ok = best.text > cfg.min_text_score
        semantic_ok = cfg.dense_candidates > 0 and best.semantic >= cfg.dense_min_sim
        if not (lexical_ok or semantic_ok):
            logger.info(
                "返信見送り: TF-IDF類似度 (%.4f) も意味類似度 (%.4f) も不足 (force=%s)",
                best.text, best.semantic, force,
            )
            return None

        # 僅差なら上位からランダムに選んで単調さを避ける
        if len(top) >= 2 and best.final - top[1].final < cfg.near_tie_threshold:
            pool = top[: cfg.near_tie_pool_size]
            floor = min(it.final for it in pool)
            weights = [max(0.05, it.final - floor + 0.15) for it in pool]
            best = self._rng.choices(pool, weights=weights, k=1)[0]
        return best

    def looking(self, x: str, u: str, reply: bool = True, force: bool = False) -> str | None:
        """過去の発言から最適な返信候補を探索"""
        logger.info("思考中: %s", x)
        sentences = self.memory["sentence"]
        if len(sentences) < 2 or not isinstance(x, str) or not x.strip():
            return None

        self._ensure_index()
        query_tokens = list(dict.fromkeys(tokenize_ngrams(x)))
        if not query_tokens:
            return None

        query_semantic = None
        if self.config.semantic_score_weight or self.config.dense_candidates:
            query_semantic = self.semantic_learner.embed_text(x)
        candidates = self._collect_candidates(query_tokens, query_semantic)
        if not candidates:
            logger.info("検索候補なし: TF-IDF・意味のどちらでも候補がありません (force=%s)", force)
            return None

        best = self._pick_best(self._score_candidates(candidates, x, u, query_semantic), force)
        if best is None:
            return None

        source, response = sentences[best.idx], sentences[best.idx + 1]
        logger.info(
            "類似: %s, idx: %d, TF-IDF: %.4f, context: %.4f, history: %.4f, "
            "style: %.4f, semantic: %.4f, final: %.3f%s, force=%s",
            source[0], best.idx, best.text, best.context, best.history, best.style,
            best.semantic, best.final, " [★good優遇]" if best.good else "", force,
        )
        logger.info("返信: %s, idx: %d", response[0], best.idx + 1)

        self.last_input_similar = source[0]
        self.last_input_similar_user = source[1]
        self.heart = best.idx + 1
        self.last_bot_base_user = response[1]
        self.last_bot_base = response[0]
        return response[0]

    # ------------------------------------------------------------------
    # 単語置換
    # ------------------------------------------------------------------
    @staticmethod
    def _diff_replacements(before_tokens: list[str], after_tokens: list[str]) -> list[tuple[str, str]]:
        """before -> after の差分から (旧, 新) の置換ペアを取り出す"""
        pairs: list[tuple[str, str]] = []
        old = new = ""
        for diff in difflib.ndiff(before_tokens, after_tokens):
            tag, content = diff[:2], diff[2:]
            if tag == "- ":
                old += content
            elif tag == "+ ":
                new += content
            elif tag == "  ":
                if (old or new) and len(old) >= 2:
                    pairs.append((old, new))
                old = new = ""
        if (old or new) and len(old) >= 2:
            pairs.append((old, new))
        return pairs

    def _is_similar_word(self, token: str, old: str) -> bool:
        if Levenshtein.normalized_similarity(token, old) >= REPLACE_CHAR_SIMILARITY:
            return True
        return self.semantic_learner.calculate_similarity(token, old) >= REPLACE_SEMANTIC_SIMILARITY

    @staticmethod
    def _person_name(speaker) -> str | None:
        """話者IDから人間の名前を取り出す。bot・システム・予約名ならNone。"""
        name = str(speaker)
        if name.startswith("!input-"):
            name = name[len("!input-"):]
        if not name or name.startswith("!") or name in BOT_SPEAKERS:
            return None
        return name

    def _register_names(self, speakers: Iterable[str]):
        """話者名と自分の名前をトークナイザーに登録する (本文中で必ず1語になる)"""
        names = {name for name in map(self._person_name, speakers) if name}
        names.update(self.my_names)
        self.tokenizer.register_words(names)

    def _remember_replacement(self, after: str, before: str):
        if not after or not before:
            return

        # 同一ペアが既にあれば削除し、最新の位置に移す
        pairs = list(zip(
            self.word_replacer_memory_after,
            self.word_replacer_memory_before,
        ))

        pairs = [
            (a, b) for a, b in pairs
            if (a, b) != (after, before)
        ]
        pairs.append((after, before))

        self.word_replacer_memory_after = [a for a, _ in pairs]
        self.word_replacer_memory_before = [b for _, b in pairs]

        self.memory["word_replacer_memory_after"] = self.word_replacer_memory_after
        self.memory["word_replacer_memory_before"] = self.word_replacer_memory_before

    def _remember_name_echo(self, result: str, base: str):
        """返信の結果 (置換後) と元の発言の差を履歴に残す。ただし名前が変わったときだけ。

        置換の結果をそのまま全部記録し直すと、一度でも誤った置換 (「まったく」→「まぃく」など) が
        起きたとき、それが新しい正規ルールとして学習されて永久に再生産される。
        名前の入れ替え (会話の間ずっと引き継ぎたいもの) だけを引き継ぐ。
        """
        names = self.tokenizer.forced_words
        if any((name in base) != (name in result) for name in names if name in base or name in result):
            self._remember_replacement(result, base)

    def _remember_self_name(self, author) -> None:
        """返信の元の発言者の名前 → 自分の名前、の置換を履歴に残す。

        新しい履歴ほど優先されるので、複数ある自分の名前は逆順に積んで、
        先頭の名前が最優先になるようにする。
        """
        name = self._person_name(author)
        if name is None:
            return
        for my_name in reversed(self.my_names):
            self._remember_replacement(my_name, name)

    def forget_replacements(self, *texts: str) -> int:
        """置換履歴から、指定した文字列を含むペアをすべて消す (汚染された履歴の掃除用)。消した件数を返す。

        例: bot.forget_replacements("まぃく", "ぃ")
        """
        pairs = [
            (a, b)
            for a, b in zip(self.word_replacer_memory_after, self.word_replacer_memory_before)
            if not any(text and (text in a or text in b) for text in texts)
        ]
        removed = len(self.word_replacer_memory_after) - len(pairs)
        self.word_replacer_memory_after = [a for a, _ in pairs]
        self.word_replacer_memory_before = [b for _, b in pairs]
        self.memory["word_replacer_memory_after"] = self.word_replacer_memory_after
        self.memory["word_replacer_memory_before"] = self.word_replacer_memory_before
        self.contextual_word_replacer._rule_cache.clear()
        self._unsaved += 1
        self.save_data()
        return removed

    def _truncate_replacer_memory(self):
        """置換履歴のサイズ制御"""
        limit = self.maximum_word_replacer_memory * (len(self.my_names) + 1)
        self.word_replacer_memory_after = self.word_replacer_memory_after[-limit:]
        self.word_replacer_memory_before = self.word_replacer_memory_before[-limit:]
        self.memory["word_replacer_memory_after"] = self.word_replacer_memory_after
        self.memory["word_replacer_memory_before"] = self.word_replacer_memory_before

    def _remember_speaker_replacements(self, speaker: str):
        """話者名の置換 (入力者 <-> 類似発言の話者 / 自分の名前) を履歴に残す"""
        if "!system" in speaker:
            return
        speaker_name = self._person_name(speaker)
        similar_name = self._person_name(self.last_input_similar_user)

        if speaker_name and similar_name:
            self._remember_replacement(speaker_name, similar_name)
        elif speaker_name:  # 類似発言は自分(bot)のもの
            for my_name in reversed(self.my_names):
                self._remember_replacement(speaker_name, my_name)
        elif similar_name:  # 今の入力は自分(bot)のもの
            for my_name in reversed(self.my_names):
                self._remember_replacement(my_name, similar_name)

    # ------------------------------------------------------------------
    # 会話の入出力
    # ------------------------------------------------------------------
    def _push_log(self, text: str, speaker: str):
        entry = [text, speaker]
        if entry != self.latest_log[-1]:
            self.latest_log.append(entry)
        if len(self.latest_log) > LOG_LIMIT:
            self.latest_log = self.latest_log[-LOG_LIMIT:]

    def _push_user_log(self, value: str):
        self.user_log.append(value)
        self.user_log.pop(0)

    def _mark_bot_turn(self):
        if "!" not in self.last_user:
            self.last_user_replied = self.last_user
            self._push_user_log("!")

    def _insert_feedback_marker(self, x: str):
        """!good / !bad を、直前に返した発言の直後に挿入する"""
        if x not in SYSTEM_MARKERS:
            return
        sentences = self.memory["sentence"]
        pos = self.heart + 1
        if pos < len(sentences) and sentences[pos][0] != x:
            sentences.insert(pos, [x, "!system"])
            self._invalidate_indexes()  # 途中挿入でidxがずれるため全再構築

    def speak_freely(self, is_active_learning: bool = True) -> str | None:
        result = self.current_voice
        self._mark_bot_turn()
        if result is None:
            return None

        self._remember_name_echo(result, self.last_bot_base)
        self._remember_self_name(self.last_bot_base_user)
        self._truncate_replacer_memory()

        self.last_previous_input_user = self.last_input_user
        self.last_previous_input_content = self.last_input_content
        self.last_bot_response = result
        self._push_log(result, "!")
        return result

    def next_speak(self, is_active_learning: bool = True) -> str | None:
        if not self.is_next_ok():
            return None

        self.heart += 1
        base_text, base_user = self.memory["sentence"][self.heart][:2]
        self.last_bot_base = base_text
        self.last_bot_base_user = base_user
        self._mark_bot_turn()

        result = self.contextual_word_replacer.execute_replace(
            base_text, self.word_replacer_memory_after, self.word_replacer_memory_before, self.tokenizer.tokenize,
            self_names=self.my_names, base_speaker=self._person_name(base_user) or "",
        )
        self._remember_name_echo(result, base_text)
        self._remember_self_name(base_user)
        self._truncate_replacer_memory()

        self.last_bot_response = result
        return result

    def receive(
        self,
        x: str,
        u: str,
        is_active_learning: bool = True,
        reply: bool = True,
        force: bool = False,
    ) -> str | None:
        if x is None or u is None:
            return None

        self.pre_heart = self.heart
        self.last_input_content = x
        self.last_input_user = u

        if "!" not in u:
            self.last_user = u
            self._push_user_log(u)
        self._register_names([u])

        if is_active_learning:
            self.learn_sentence(x, u)
        self._insert_feedback_marker(x)
        self._push_log(x, u)

        result = self.looking(x, u, reply=reply, force=force)
        if result is None:
            self.current_voice = None
            return None

        self._remember_replacement(x, self.last_input_similar)
        self._remember_speaker_replacements(u)
        # 返信の元の発言者が自分の名前を名乗っていたら、自分の名前に直す
        # (これを speak_freely まで待つと、最初の1回目の返信には間に合わない)
        self._remember_self_name(self.last_bot_base_user)
        self._truncate_replacer_memory()

        result = self.contextual_word_replacer.execute_replace(
            result, self.word_replacer_memory_after, self.word_replacer_memory_before, self.tokenizer.tokenize,
            self_names=self.my_names, base_speaker=self._person_name(self.last_bot_base_user) or "",
        )
        result = result.replace("[YOU]", self.last_user)
        result = result.replace("[ME]", self.settings["myname"])

        self.current_voice = result
        logger.info("座標: %d / ログ: %s / 心の声: %s", self.heart, self.user_log, result)
        return result

    # ------------------------------------------------------------------
    # ゲッター
    # ------------------------------------------------------------------
    def get_settings(self) -> dict:
        return self.settings

    def get_memory(self) -> dict:
        return self.memory

    def get_current_voice(self) -> str | None:
        return self.current_voice

    def get_last_bot_response(self) -> str:
        return self.last_bot_response

    def get_last_user(self) -> str:
        return self.last_user


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Himawaria 動作テスト")
    parser.add_argument("directory", nargs="?", default="kasen", help="記憶ディレクトリ")
    parser.add_argument("--log-file", help="ログをこのファイルにも書き出す")
    parser.add_argument("--quiet", action="store_true", help="ログをWARNING以上だけにする")
    args = parser.parse_args()

    setup_logging(logging.WARNING if args.quiet else logging.INFO, args.log_file)
    bot = Himawaria(args.directory)

    print("--- 初期学習中 ---")
    bot.update_tfidf_index()

    print("\n--- 会話テスト ---")
    print("応答:", bot.receive("こんにちは", "ユーザー", is_active_learning=True))

    print("\n--- 単語の演算テスト ---")
    for word, score in bot.semantic_learner.arithmetic(
        positive=["私", "名前"], top_n=3
    ):
        print(f"  単語: {word} (意味類似度: {score:.4f})")