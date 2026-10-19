import json
import random
import pickle
import math
import re
import difflib
from collections import Counter
from rapidfuzz.distance import Levenshtein


# ---------------------------------------------------------------------------
# 検索・会話処理の調整値
# 動作を調整するときは、まずこの定数を変更し、looking() 内に数値を散在させない。
# ---------------------------------------------------------------------------
BM25_K1 = 1.5
BM25_B = 0.75
MIN_BM25_CANDIDATE_SCORE = 0.01
GOOD_TAG_MULTIPLIER = 1.15
TEXT_SCORE_WEIGHT = 0.72
CONTEXT_SCORE_WEIGHT = 0.0
HISTORY_SCORE_WEIGHT = 0.10
GOOD_TAG_BONUS = 0.04
# キャラクター口調は補助情報。入力内容の一致より強くならないよう、ごく小さくする。
CHARA_STYLE_WEIGHT = 0.02
NEAR_TIE_THRESHOLD = 0.035
NEAR_TIE_POOL_SIZE = 3
HISTORY_DECAY_WEIGHTS = (1.0, 0.5, 0.25, 0.125, 0.0625)


class BM25Searcher:
    """文字n-gramを使った簡易BM25検索器。fit()で索引を作り、get_score()で採点する。"""
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.doc_len = []
        self.avgdl = 0
        self.doc_freqs = []
        self.idf = {}
        self.corpus_size = 0
        self.docs_tokens = []

    @staticmethod
    def tokenize(text, n=2):
        if not text:
            return []

        # 2文字以上の同じ文字の連続は「2文字」に圧縮する（例: "..." や "...." -> "..", "ーーー" -> "ーー"）
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)

        tokens = []
        parts = re.split(r'(\d+)', text)
        for part in parts:
            if not part:
                continue
            if part.isdigit():
                tokens.append(part)
            else:
                if len(part) < n:
                    tokens.append(part)
                else:
                    tokens.extend([part[i:i+n] for i in range(len(part) - n + 1)])
        return tokens

    def fit(self, corpus):
        """記憶データ全体からBM25インデックスを再構築する。"""
        # 同じインスタンスでfitを再実行しても古い統計が残らないよう初期化
        self.doc_len = []
        self.doc_freqs = []
        self.idf = {}
        self.docs_tokens = []
        self.avgdl = 0.0
        self.corpus_size = len(corpus)

        if self.corpus_size == 0:
            return

        self.docs_tokens = [self.tokenize(doc) for doc in corpus]
        self.doc_len = [len(tokens) for tokens in self.docs_tokens]
        self.avgdl = sum(self.doc_len) / self.corpus_size

        df = Counter()
        for tokens in self.docs_tokens:
            frequencies = Counter(tokens)
            self.doc_freqs.append(frequencies)
            df.update(frequencies.keys())

        self.idf = {
            token: math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0)
            for token, freq in df.items()
        }

    def get_score(self, query_tokens, index):
        """指定したインデックスの記憶とのBM25スコアを算出"""
        score = 0.0
        doc_tokens = self.doc_freqs[index]
        d_len = self.doc_len[index]

        if self.avgdl == 0:
            return 0.0

        for token in query_tokens:
            if token not in doc_tokens:
                continue
            idf = self.idf.get(token, 0.0)
            tf = doc_tokens[token]
            # BM25 Core Formula
            num = tf * (self.k1 + 1)
            den = tf + self.k1 * (1 - self.b + self.b * (d_len / self.avgdl))
            score += idf * (num / den)

        return score

class NgramTokenizer:
    """文字の結合度を学習し、単語置換処理で使うトークナイザー。"""
    def __init__(self):
        self.char_counts = Counter()
        self.bigram_counts = Counter()
        self.forced_words = set()
        
    def train(self, corpus_list):
        """1文字と2文字の出現頻度をカウント"""
        for sentence in corpus_list:
            if not sentence:
                continue
            self.char_counts.update(sentence)
            bigrams = [sentence[i:i+2] for i in range(len(sentence)-1)]
            self.bigram_counts.update(bigrams)

    def register_words(self, words_list):
        """分割してほしくない単語を辞書として登録する"""
        self.forced_words.update(words_list)

    def _get_cohesion_score(self, c1, c2):
        """2文字の結合度スコア（条件付き確率）"""
        bigram = c1 + c2
        if self.char_counts[c1] == 0 or self.bigram_counts[bigram] == 0:
            return 0.0
        return self.bigram_counts[bigram] / self.char_counts[c1]

    def tokenize(self, text, drop_threshold=0.3):
        """結合度スコアがしきい値より低いタイミングで区切る"""
        if len(text) <= 1:
            return [text]
            
        words = []
        current_word = text[0]
        
        for i in range(len(text) - 1):
            c1, c2 = text[i], text[i+1]
            candidate = current_word + c2
            
            is_part_of_forced_word = any(
                fw.startswith(candidate) for fw in self.forced_words
            )
            
            if is_part_of_forced_word:
                current_word = candidate
            else:
                score = self._get_cohesion_score(c1, c2)
                if score < drop_threshold:
                    words.append(current_word)
                    current_word = c2
                else:
                    current_word += c2
                
        if current_word:
            words.append(current_word)
            
        return words
    
    def save(self, file_path):
        """学習済みの統計データをファイルに保存する"""
        self_to_save = {
            'char_counts': self.char_counts,
            'bigram_counts': self.bigram_counts
        }
        with open(file_path, 'wb') as f:
            pickle.dump(self_to_save, f)

    def load(self, file_path):
        """保存されたファイルから統計データを復元する"""
        with open(file_path, 'rb') as f:
            loaded_self = pickle.load(f)
        self.char_counts = loaded_self['char_counts']
        self.bigram_counts = loaded_self['bigram_counts']
        print(f"🚀 モデルを正常に読み込みました: {file_path}")

class Himawaria:
    """会話記憶の検索・学習・応答を管理するクラス。

    memory.json の sentence は [発言, 話者] のペアを順に保持する。
    looking() は候補検索と順位付けを行い、receive() は入力から応答までを管理する。
    """
    def __init__(self, directory, maximum_word_replacer_memory=128, min_similarity_threshold=0.45):
        self.direc = directory
        self.maximum_word_replacer_memory = maximum_word_replacer_memory
        self.min_similarity_threshold = min_similarity_threshold

        self.memory = None
        self.settings = None
        self.heart = None
        self.last_bot_response = ""
        self.last_bot_base = ""
        self.last_input_content = ""
        self.last_input_similar = ""
        self.last_bot_base_user = ""
        self.last_input_user = ""
        self.last_input_similar_user = ""
        self.pre_heart = 0
        self.last_user = "あんた"
        self.last_user_bot_replied = "あんた"
        self.current_voice = None
        self.user_log = [None] * 10
        self.word_replacer_memory_after = []
        self.word_replacer_memory_before = []
        self.last_previous_input_user = ""
        self.rate = 1.0
        self.latest_log = [["init", "init"]]

        try:
            with open(self.direc+"/memory.json", "r", encoding="utf8") as f:
                self.memory = json.load(f)
            with open(self.direc+"/settings.json", "r", encoding="utf8") as f:
                self.settings = json.load(f)
        except:
            with open(self.direc+"/memory_backup.json", "r", encoding="utf8") as f:
                self.memory = json.load(f)
            with open(self.direc+"/settings.json", "r", encoding="utf8") as f:
                self.settings = json.load(f)
            with open(self.direc+"/memory.json", "w", encoding="utf8") as f:
                json.dump(self.memory, f, ensure_ascii=False, indent=4, sort_keys=True, separators=(',', ': '))
        self.heart = len(self.memory["sentence"]) - 1

        try:
            with open(self.direc+"/chara_style.txt", "r", encoding="utf8") as f:
                self.chara_style = f.read()
        except:
            with open(self.direc+"/chara_style.txt", "w", encoding="utf8") as f:
                f.write("")
                self.chara_style = ""

        with open(self.direc+"/memory_backup.json", "w", encoding="utf8") as f:
            json.dump(self.memory, f, ensure_ascii=False, indent=4, sort_keys=True, separators=(',', ': '))

        self.memory.setdefault("word_replacer_memory_after", [])
        self.memory.setdefault("word_replacer_memory_before", [])
        
        self.word_replacer_memory_after = self.memory["word_replacer_memory_after"]
        self.word_replacer_memory_before = self.memory["word_replacer_memory_before"]

        self.tokenizer = NgramTokenizer()
        try:
            self.tokenizer.load(self.direc+"/tokenizer.model")
        except:
            for sen in self.memory["sentence"]:
                self.tokenizer.train([sen[0]])
                self.tokenizer.train([sen[1]])
        self.heart = random.randint(0, max(0, len(self.memory["sentence"]) - 1))

        """
        sentences = self.memory["sentence"]
        corpus = []
        
        for i in range(len(sentences)):
            curr_text, speaker = sentences[i][0], sentences[i][1]
            curr_tag = "[!]" if speaker == "!" else f"[{speaker}]"
            # 余計な過去ログの連結をやめ、その発言単体をドキュメントにする
            corpus.append(f"{curr_tag}{curr_text}")

        self.bm25 = BM25Searcher(k1=BM25_K1, b=BM25_B)
        self.bm25.fit(corpus)
        print(f"NgramTokenizer: {self.tokenizer.tokenize("みんなが幼児化しちゃった件")}")
        print(f"BM25Searcher: {self.bm25.tokenize("みんなが幼児化しちゃった件")}")
        """
        
    # ------------------------------------------------------------------
    # 学習・保存
    # ------------------------------------------------------------------
    def learnSentence(self, x, u, save=True, directLearning=False):
        if u not in self.settings["mynames"].split("|") and directLearning and u not in ["!input", "!output", "!system"]:
            u = "!input-"+u

        if u in self.settings["mynames"].split("|"):
            self.memory["sentence"].append([x, "!output"])
        else:
            self.memory["sentence"].append([x, u])
        self.tokenizer.train([x, u])

        if len(self.memory["sentence"]) >= 1600000:
            while len(self.memory["sentence"]) >= 1600000:
                del self.memory["sentence"][0]
                if self.heart >= 5 or self.pre_heart >= 5:
                    self.heart -= 1
                    self.pre_heart -= 1
        if save:
            self.saveData()

    def saveData(self):
        with open(self.direc+"/memory.json", "w", encoding="utf8") as f:
            json.dump(self.memory, f, ensure_ascii=False, indent=4, sort_keys=True, separators=(',', ': '))
        self.tokenizer.save(self.direc+"/tokenizer.model")

    def evalute(self):
        flag1 = False
        flag2 = False
        if self.memory["sentence"][-1][0] not in ["!bad", "!good"]:
            if self.heart + 1 < len(self.memory["sentence"]) - 1:
                if self.memory["sentence"][self.heart+1][0] == "!good":
                    flag1 = True
            if flag1:
                print("このメッセージは良い")
                self.learnSentence("!good", "!system")

        if self.memory["sentence"][-1][0] not in ["!bad", "!good"] and not flag1:
            if self.heart + 1 < len(self.memory["sentence"]) - 1:
                if self.memory["sentence"][self.heart+1][0] == "!bad":
                    flag2 = True
            if flag2:
                print("このメッセージは悪い")
                self.learnSentence("!bad", "!system")

    def isNextOk(self):
        if len(self.memory["sentence"]) - 1 <= self.heart + 1:
            return False
        next_sen = self.memory["sentence"][self.heart + 1]
        curr_sen = self.memory["sentence"][self.heart]
        return (self.last_input_content != next_sen[0] and
                self.last_bot_response != next_sen[0] and
                next_sen[1] == curr_sen[1] and
                next_sen[1] != "!" and
                "!system" not in next_sen[1])

    # ------------------------------------------------------------------
    # 応答後の単語置換
    # ------------------------------------------------------------------
    def replaceWords(self, x, inputs, inputsHeart):
        """
        単語置換処理: 助詞や短いトークンの破壊的・誤認置換を抑制し、安全な置換を実施
        """
        replacements = []
        used = []
        w3 = self.tokenizer.tokenize(x, drop_threshold=0.4)
        
        for i in range(len(inputs)):
            if not inputs[i] or not inputsHeart[i]:
                continue
            w1 = self.tokenizer.tokenize(inputs[i], drop_threshold=0.4)
            w2 = self.tokenizer.tokenize(inputsHeart[i], drop_threshold=0.4)

            diffs = list(difflib.ndiff(w2, w1))
            old, new = "", ""
            for diff in diffs:
                tag, content = diff[:2], diff[2:]
                if tag == "- ":
                    old += content
                elif tag == "+ ":
                    new += content
                elif tag == "  ":
                    if old or new:
                        if old and new and len(old) >= 2:
                            replacements.append((old, new))
                        old, new = "", ""
            if (old or new) and len(old) >= 2:
                replacements.append((old, new))
                
        res_tokens = list(w3)
        for idx, token in enumerate(res_tokens):
            if len(token) <= 1:
                continue
            for old, new in reversed(replacements):
                sim = Levenshtein.normalized_similarity(token, old)
                if sim >= 0.85:
                    print(f"単語置換: {token} ({old}) => {new}")
                    res_tokens[idx] = new
                    break

        return "".join(res_tokens)

    def isAvailable(self, d, b, type=1):
        if type == 0:
            flag = False
            for i in range(1):
                if b + 2 + i < len(self.memory["sentence"]) - 1:
                    if self.memory["sentence"][b+2+i][0] == "!good":
                        flag = True
                        break
            for i in range(1):
                if b + 2 + i < len(self.memory["sentence"]) - 1:
                    if self.memory["sentence"][b+2+i][0] == "!bad":
                        flag = False
                        break
            return flag and d >= 0.6
        elif type == 1:
            flag = True
            for i in range(1):
                if b + 2 + i < len(self.memory["sentence"]) - 1:
                    if self.memory["sentence"][b+2+i][0] == "!bad":
                        flag = False
                        break
            return flag

    # ------------------------------------------------------------------
    # 検索候補の生成・採点
    # ------------------------------------------------------------------
    def update_bm25_index(self):
        """各発言をシンプルにインデックス化する"""
        sentences = self.memory["sentence"]
        corpus = []

        for i in range(len(sentences)):
            curr_text, speaker = sentences[i][0], sentences[i][1]
            curr_tag = "[!]" if speaker == "!" else f"[{speaker}]"
            # 余計な過去ログの連結をやめ、その発言単体をドキュメントにする
            corpus.append(f"{curr_tag}{curr_text}")

        self.bm25 = BM25Searcher(k1=BM25_K1, b=BM25_B)
        self.bm25.fit(corpus)

    def _is_duplicate(self, reply, last_input, last_bot_resp, last_bot_base):
        """短文や記号（「？」など）が重複判定で弾かれるのを防ぐヘルパー"""
        if len(reply) <= 2:
            return reply == last_bot_resp or reply == last_bot_base
        
        if Levenshtein.normalized_similarity(reply, last_input) >= 0.85:
            return True
        if Levenshtein.normalized_similarity(reply, last_bot_resp) >= 0.85:
            return True
        if Levenshtein.normalized_similarity(reply, last_bot_base) >= 0.85:
            return True
            
        return False

    def _has_good_tag(self, idx):
        """指定位置の1つ下、または2つ下（返信のすぐ下）に !good があるか判定"""
        total_len = len(self.memory["sentence"])
        for offset in (1, 2):
            if idx + offset < total_len:
                if self.memory["sentence"][idx + offset][0] == "!good":
                    return True
        return False

    @staticmethod
    def _normalize_scores(values):
        """スコア列を0〜1に正規化する。同点しかない場合は全件0を返す。"""
        if not values:
            return []
        low, high = min(values), max(values)
        if high - low <= 1e-12:
            return [0.0] * len(values)
        return [(value - low) / (high - low) for value in values]

    def looking(self, x, u, reply=True, force=False):
        """現在の入力に近い過去発言を検索し、その次の発言を返す。"""
        print(f"思考中: {x}")
        sentences = self.memory.get("sentence", [])
        total_len = len(sentences)
        if total_len < 2 or not isinstance(x, str) or not x.strip():
            return None

        # learnSentence() が先に呼ばれる設計なので、通常は件数差で再構築される。
        # fit() 自体も安全に再実行できるよう上で修正済み。
        if (not hasattr(self, "bm25") or
                self.bm25.corpus_size != total_len or
                len(self.bm25.doc_freqs) != total_len):
            self.update_bm25_index()

        # BM25のクエリ側で同じn-gramを重複加点しない。
        # 日本語の短い文字列では重複トークンがスコアを不自然に押し上げるため、
        # まずユニークなn-gramで検索する。
        tokens_query = list(dict.fromkeys(BM25Searcher.tokenize(x)))
        if not tokens_query:
            return None

        def is_system_or_marker(entry):
            text, speaker = entry[0], str(entry[1])
            return (
                "!system" in speaker or
                text in ("!bad", "!good") or
                speaker == "!"
            )

        def valid_pair(idx):
            """idxの発言と直後の返信が検索候補として有効か判定。"""
            if idx < 0 or idx + 1 >= total_len:
                return False
            source = sentences[idx]
            response = sentences[idx + 1]
            response_text, response_speaker = response[0], str(response[1])

            if is_system_or_marker(response):
                return False
            if response_text is None or not str(response_text).strip():
                return False
            if response_speaker == str(source[1]):
                return False
            if self._is_duplicate(
                str(response_text),
                str(self.last_input_content or ""),
                str(self.last_bot_response or ""),
                str(self.last_bot_base or "")
            ):
                return False
            return True

        # 現在の入力に関連する過去発言を検索。
        # BM25は一部のn-gramだけが一致しても高得点になりうるため、
        # 入力n-gramのどれだけが候補発言に含まれるかも確認する。
        # force=Trueでも、この関連性フィルターは迂回させない。
        candidates = []
        min_query_matches = min(2, len(set(tokens_query)))
        for idx in range(total_len - 1):
            if not valid_pair(idx):
                continue

            doc_tokens = set(self.bm25.docs_tokens[idx])
            query_token_set = set(tokens_query)
            matched_tokens = query_token_set.intersection(doc_tokens)
            match_count = len(matched_tokens)
            coverage = match_count / max(1, len(query_token_set))

            # 長い入力が「ちゃ」など1個の偶然の一致で通過するのを防ぐ。
            # 2文字以下の入力は1トークンしかないため、完全一致相当の条件で判定。
            if match_count < min_query_matches:
                continue
            if len(query_token_set) >= 4 and coverage < 0.20:
                continue

            score = self.bm25.get_score(tokens_query, idx)
            if self._has_good_tag(idx):
                score *= GOOD_TAG_MULTIPLIER
            if score > MIN_BM25_CANDIDATE_SCORE:
                candidates.append((score, idx))

        # 関連する候補が見つからない場合、force=Trueでも無関係な記憶を
        # 全件候補に戻さない。ゼロスコアの候補を文脈・goodタグだけで選ぶと、
        # 「？」のような入力に対して無関係な返信を生成してしまう。
        if not candidates:
            print(f"検索候補なし: 入力とのBM25関連度が閾値以下です (force={force})")
            return None

        # ライブ履歴から「現在の入力より前」のユーザー発言のみを取得。
        # latest_logには通常 [発言, 話者] が格納される。
        live_history = []
        live_log = getattr(self, "latest_log", [])
        prior_log = live_log[:-1] if live_log and live_log[-1] == [x, u] else live_log
        bot_names = {"!", "!output", "bot", "init"}
        for item in reversed(prior_log):
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            h_text, h_speaker = item[0], str(item[1])
            if not isinstance(h_text, str) or not h_text.strip():
                continue
            if h_speaker in bot_names or "!system" in h_speaker or h_speaker.startswith("!input-"):
                continue
            live_history.append(h_text)
            if len(live_history) >= 5:
                break
        if not live_history:
            live_history = []

        live_history_tokens = [BM25Searcher.tokenize(t) for t in live_history]
        decay_weights = HISTORY_DECAY_WEIGHTS

        # 候補ごとに、候補発言より前にあるユーザー発言列との一致を測る。
        def previous_user_history(target_idx, limit=5):
            history = []
            for j in range(target_idx - 1, -1, -1):
                entry = sentences[j]
                if is_system_or_marker(entry):
                    continue
                speaker = str(entry[1])
                if speaker in bot_names or speaker.startswith("!output"):
                    continue
                if "!system" in speaker or speaker.startswith("!input-"):
                    continue
                text = entry[0]
                if isinstance(text, str) and text.strip():
                    history.append(text)
                    if len(history) >= limit:
                        break
            return history

        chara_style_query = BM25Searcher.tokenize(self.chara_style)
        scored = []
        for text_score, idx in candidates:
            # 文脈スコアは入力単体より弱く扱い、極端な加点を避ける。
            context_tokens = []
            # 候補直前の発言を使い、存在しないself.messagesには依存しない。
            for j in range(max(0, idx - 4), idx):
                previous_text = sentences[j][0]
                previous_speaker = str(sentences[j][1])
                if (isinstance(previous_text, str) and previous_text.strip()
                        and previous_speaker not in bot_names
                        and "!system" not in previous_speaker
                        and previous_text not in ("!good", "!bad")):
                    context_tokens.extend(BM25Searcher.tokenize(previous_text))
            context_score = self.bm25.get_score(context_tokens, idx) if context_tokens else 0.0

            memory_history = previous_user_history(idx, limit=5)
            history_score = 0.0
            matched = min(len(live_history_tokens), len(memory_history), len(decay_weights))
            for k in range(matched):
                query_tokens = live_history_tokens[k]
                if not query_tokens:
                    continue
                # 新しい履歴ほど大きな重み。候補の直前から順に対応させる。
                memory_pos = idx - 1
                seen = 0
                while memory_pos >= 0 and seen <= k:
                    entry = sentences[memory_pos]
                    if not is_system_or_marker(entry):
                        speaker = str(entry[1])
                        if (speaker not in bot_names and not speaker.startswith("!input-")
                                and "!system" not in speaker):
                            if seen == k:
                                break
                            seen += 1
                    memory_pos -= 1
                if memory_pos >= 0:
                    history_score += self.bm25.get_score(query_tokens, memory_pos) * decay_weights[k]

            chara_style_score = self.bm25.get_score(chara_style_query, idx+1)
            scored.append({
                "idx": idx,
                "text": text_score,
                "context": context_score,
                "history": history_score,
                "good": self._has_good_tag(idx),
                "chara_style": chara_style_score,
            })

        if not scored:
            return None

        # 各特徴量はBM25値の尺度が異なるため、候補集合内で0〜1に正規化してから合成する。
        # 注意: この正規化値は候補間の相対値であり、絶対的な確信度ではない。
        text_norm = self._normalize_scores([item["text"] for item in scored])
        context_norm = self._normalize_scores([item["context"] for item in scored])
        history_norm = self._normalize_scores([item["history"] for item in scored])
        chara_style_norm = self._normalize_scores([item["chara_style"] for item in scored])

        for i, item in enumerate(scored):
            # 入力との一致を主軸にし、文脈・履歴・口調は補助的に加点する。
            # !good は既存の品質優遇機構として固定ボーナスを与える。
            item["final"] = (
                TEXT_SCORE_WEIGHT * text_norm[i]
                + CONTEXT_SCORE_WEIGHT * context_norm[i]
                + HISTORY_SCORE_WEIGHT * history_norm[i]
                + (GOOD_TAG_BONUS if item["good"] else 0.0)
                + CHARA_STYLE_WEIGHT * chara_style_norm[i]
            )

        scored.sort(key=lambda item: item["final"], reverse=True)

        # 正規化値は候補間の相対値にすぎないため、最終スコアだけでなく
        # 入力そのものとのBM25関連度を必ず確認する。force=Trueでもここは迂回させない。
        # これにより、文脈スコアや !good の加点だけで無関係な返信が選ばれるのを防ぐ。
        if scored[0]["text"] <= 0.01:
            print(
                f"返信見送り: 入力とのBM25スコアが不足 "
                f"(score={scored[0]['text']:.4f}, force={force})"
            )
            return None

        # 上位候補がほぼ同点なら、上位3件の範囲で品質寄りの重み付き抽選。
        best = scored[0]
        if (len(scored) >= 2 and
                best["final"] - scored[1]["final"] < NEAR_TIE_THRESHOLD):
            pool = scored[:min(NEAR_TIE_POOL_SIZE, len(scored))]
            floor = min(item["final"] for item in pool)
            weights = [max(0.05, item["final"] - floor + 0.15) for item in pool]
            chosen = random.choices(pool, weights=weights, k=1)[0]
            # 低スコアの候補を選んだ場合も、選択された候補の値を一貫して使う。
            best = chosen

        best_idx = best["idx"]
        response = sentences[best_idx + 1]
        is_good_mark = " [★good優遇]" if best["good"] else ""
        print(
            f"類似: {sentences[best_idx][0]}, idx: {best_idx}, "
            f"BM25: {best['text']:.4f}, context: {best['context']:.4f}, "
            f"history: {best['history']:.4f}, style: {best['chara_style']:.4f}, "
            f"FinalScore: {best['final']:.3f}"
            f"{is_good_mark}, force={force}"
        )
        print(f"返信: {response[0]}, idx: {best_idx + 1}")

        self.last_input_similar = sentences[best_idx][0]
        self.last_input_similar_user = sentences[best_idx][1]
        self.heart = best_idx + 1
        self.last_bot_base_user = response[1]
        self.last_bot_base = response[0]
        return response[0]


    # ------------------------------------------------------------------
    # 会話フロー
    # ------------------------------------------------------------------
    def record(self):
        if self.current_voice:
            self.learnSentence(self.current_voice, "!")
            self.evalute()
        
    def speakFreely(self, is_active_learning=True):
        result = self.current_voice
        if "!" not in self.last_user:
            self.last_userReplied = self.last_user
        self.user_log.append("!")
        self.user_log.pop(0)
        if result is not None:
            self.word_replacer_memory_after.append(result)
            self.word_replacer_memory_before.append(self.last_bot_base)
            if self.last_bot_base_user in ["!", "!output"]:
                for myname in self.settings["mynames"].split("|"):
                    self.word_replacer_memory_after.append(myname)
                    self.word_replacer_memory_before.append(self.last_bot_base_user)
            if len(self.word_replacer_memory_after) > self.maximum_word_replacer_memory:
                self.word_replacer_memory_after = self.word_replacer_memory_after[-self.maximum_word_replacer_memory:]
            if len(self.word_replacer_memory_before) > self.maximum_word_replacer_memory:
                self.word_replacer_memory_before = self.word_replacer_memory_before[-self.maximum_word_replacer_memory:]
            self.memory["word_replacer_memory_after"] = self.word_replacer_memory_after
            self.memory["word_replacer_memory_before"] = self.word_replacer_memory_before

        self.last_previous_input_user = self.last_input_user
        self.last_previous_input_content = self.last_input_content
        self.last_bot_response = result
        if [result, "!"] != self.latest_log[-1]:
            self.latest_log.append([result, "!"])
        if len(self.latest_log) > 9:
            self.latest_log = self.latest_log[-9:]
        return result

    def nextSpeak(self, is_active_learning=True):
        if self.isNextOk():
            self.heart += 1
            result = self.memory["sentence"][self.heart][0]
            self.last_bot_base = result
            self.last_bot_base_user = self.memory["sentence"][self.heart][1]
            if "!" not in self.last_user:
                self.last_userReplied = self.last_user
            self.user_log.append("!")
            self.user_log.pop(0)
            if result is not None:
                result = self.replaceWords(result, self.word_replacer_memory_after, self.word_replacer_memory_before)

                self.word_replacer_memory_after.append(result)
                self.word_replacer_memory_before.append(self.last_bot_base)
                for myname in reversed(self.settings["mynames"].split("|")):
                    self.word_replacer_memory_after.append(myname)
                    self.word_replacer_memory_before.append(self.last_bot_base_user)
                limit = self.maximum_word_replacer_memory * (len(self.settings["mynames"].split("|")) + 1)
                self.word_replacer_memory_after = self.word_replacer_memory_after[-limit:]
                self.word_replacer_memory_before = self.word_replacer_memory_before[-limit:]
                self.memory["word_replacer_memory_after"] = self.word_replacer_memory_after
                self.memory["word_replacer_memory_before"] = self.word_replacer_memory_before

            self.last_bot_response = result
            return result
        else:
            return None

    def receive(self, x, u, is_active_learning=True, reply=True, force=False):
        if x is None or u is None:
            return None
        
        self.pre_heart = self.heart
        self.last_input_content = x
        self.last_input_user = u
        if "!" not in u:
            self.last_user = u
            self.user_log.append(u)
            self.user_log.pop(0)
        
        if is_active_learning:
            self.learnSentence(x, u)
            if x == "!bad" and self.memory["sentence"][self.heart+1][0] != "!bad":
                self.memory["sentence"].insert(self.heart+1, ["!bad", "!"])
            if x == "!good" and self.memory["sentence"][self.heart+1][0] != "!good":
                self.memory["sentence"].insert(self.heart+1, ["!good", "!"])
        if [x, u] != self.latest_log[-1]:
            self.latest_log.append([x, u])
        if len(self.latest_log) > 9:
            self.latest_log = self.latest_log[-9:]
        result = self.looking(x, u, force=force, reply=reply)
        
        if result is None:
            self.current_voice = None
            return None

        self.word_replacer_memory_after.append(x)
        self.word_replacer_memory_before.append(self.last_input_similar)

        if "!system" not in u:
            if u not in ["!", "!output"] and self.last_input_similar_user not in ["!", "!output"]:
                self.word_replacer_memory_after.append(u)
                self.word_replacer_memory_before.append(self.last_input_similar_user)
            elif self.last_input_similar_user in ["!", "!output"] and u not in ["!", "!output"]:
                for myname in self.settings["mynames"].split("|"):
                    self.word_replacer_memory_after.append(u)
                    self.word_replacer_memory_before.append(myname)
            elif u in ["!", "!output"] and self.last_input_similar_user not in ["!", "!output"]:
                for myname in self.settings["mynames"].split("|"):
                    self.word_replacer_memory_after.append(myname)
                    self.word_replacer_memory_before.append(self.last_input_similar_user)
        limit = self.maximum_word_replacer_memory * (len(self.settings["mynames"].split("|")) + 1)
        self.word_replacer_memory_after = self.word_replacer_memory_after[-limit:]
        self.word_replacer_memory_before = self.word_replacer_memory_before[-limit:]

        result = self.replaceWords(result, self.word_replacer_memory_after, self.word_replacer_memory_before)
        result = result.replace("[YOU]", self.last_user)

        self.memory["word_replacer_memory_after"] = self.word_replacer_memory_after
        self.memory["word_replacer_memory_before"] = self.word_replacer_memory_before
            
        self.current_voice = result
        print(f"座標: {self.heart}")
        print(f"ログ: {self.user_log}")
        print(f"心の声: {result}")
        return result

    def get_settings(self):
        return self.settings
    
    def get_memory(self):
        return self.memory
    
    def get_current_voice(self):
        return self.current_voice
    
    def get_last_bot_response(self):
        return self.last_bot_response
    
    def get_last_user(self):
        return self.last_user


if __name__ == "__main__":
    a = Himawaria("kasen")
    print(a.receive("こんにちは", "ユーザー", True))
    print(a.receive("調子はどう？", "ユーザー", True))