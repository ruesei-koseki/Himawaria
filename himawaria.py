import json
import random
import pickle
from rapidfuzz.distance import Levenshtein
from collections import Counter
import difflib
import math
import re

class BM25Searcher:
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
        if len(text) < n:
            return [text] if text else []
        return [text[i:i+n] for i in range(len(text) - n + 1)]

    def calc_term_score(self, tf, doc_len, idf):
        if self.avgdl == 0:
            return 0.0
        num = tf * (self.k1 + 1)
        den = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avgdl))
        return idf * (num / den)

    def fit(self, corpus):
        self.corpus_size = len(corpus)
        if self.corpus_size == 0:
            return

        self.docs_tokens = [self.tokenize(doc) for doc in corpus]
        self.doc_len = [len(tokens) for tokens in self.docs_tokens]
        self.avgdl = sum(self.doc_len) / self.corpus_size if self.corpus_size > 0 else 0

        df = Counter()
        for tokens in self.docs_tokens:
            frequencies = Counter(tokens)
            self.doc_freqs.append(frequencies)
            for token in frequencies.keys():
                df[token] += 1

        for token, freq in df.items():
            self.idf[token] = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0)

    def get_score(self, query_tokens, index):
        score = 0.0
        doc_tokens = self.doc_freqs[index]
        d_len = self.doc_len[index]

        for token in query_tokens:
            if token in doc_tokens:
                idf = self.idf.get(token, 0.0)
                tf = doc_tokens[token]
                score += self.calc_term_score(tf, d_len, idf)

        return score


class BM25WordReplacer:
    """直近履歴 (before/after) と BM25 を活用した安全な単語置換エンジン"""
    def __init__(self, bm25_searcher=None):
        self.bm25 = bm25_searcher or BM25Searcher()

    def is_protected_token(self, text):
        """数字のみ（IDや数値）、またはコマンド系は置換から保護する"""
        if text.isdigit():  # 数字のみの文字列（ID等）
            return True
        return False

    def extract_replacement_pairs(self, before_text, after_text, min_len=2):
        """
        before(元文) と after(入力/変換後文) の差分から置換ペア (old, new) を抽出
        ※ 数字だけの部分置換ペアの生成を防ぐ
        """
        matcher = difflib.SequenceMatcher(None, before_text, after_text)
        pairs = []
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'replace':
                old_word = before_text[i1:i2].strip()
                new_word = after_text[j1:j2].strip()
                
                # 長さチェック
                if len(old_word) < min_len or len(new_word) < min_len:
                    continue
                
                # 数字のみの置換ペア（IDの部分書き換えの原因）は除外
                if old_word.isdigit() or new_word.isdigit():
                    continue

                pairs.append((old_word, new_word))
        return pairs

    def _calc_text_context_score(self, target_text, context_tokens):
        """特定のテキストが指定のトークン群（文脈）とどれだけBM25スコアで共起するか評価"""
        target_tokens = self.bm25.tokenize(target_text)
        if not target_tokens or not context_tokens:
            return 0.0

        score = 0.0
        doc_len = len(target_tokens)
        avgdl = self.bm25.avgdl if self.bm25.avgdl > 0 else 1.0

        tf_dict = Counter(target_tokens)
        for token in context_tokens:
            if token in tf_dict:
                tf = tf_dict[token]
                idf = self.bm25.idf.get(token, 0.1)
                num = tf * (self.k1 if hasattr(self, 'k1') else 1.5 + 1)
                den = tf + (self.k1 if hasattr(self, 'k1') else 1.5) * (1 - 0.75 + 0.75 * (doc_len / avgdl))
                score += idf * (num / den)

        return score

    def replace_and_validate(self, response_candidate, memory_before_list, memory_after_list, tolerance=0.15):
        if not memory_before_list or not memory_after_list:
            return response_candidate

        # 1. コマンド文言のガード（!command から始まるメッセージは置換処理自体をスキップ）
        if response_candidate.startswith("!"):
            return response_candidate

        # 2. 直近履歴から置換可能なペアを全抽出
        candidate_pairs = []
        for b_text, a_text in zip(memory_before_list, memory_after_list):
            pairs = self.extract_replacement_pairs(b_text, a_text)
            candidate_pairs.extend(pairs)

        if not candidate_pairs:
            return response_candidate

        # 3. IDなどの長い数字列（8桁以上）を正規表現で保護（一時的なプレースホルダーに退避）
        protected_ids = re.findall(r'\d{8,}', response_candidate)
        s_current = response_candidate
        for idx, pid in enumerate(protected_ids):
            s_current = s_current.replace(pid, f"__PROTECTED_ID_{idx}__")

        # 4. 置換の適用処理
        recent_context_text = " ".join(memory_after_list)
        context_tokens = self.bm25.tokenize(recent_context_text)

        for old_word, new_word in candidate_pairs:
            if old_word in s_current and old_word != new_word:
                s_after_temp = s_current.replace(old_word, new_word, 1)

                score_before = self._calc_text_context_score(s_current, context_tokens)
                score_after = self._calc_text_context_score(s_after_temp, context_tokens)

                is_valid = False
                if score_before > 0:
                    if score_after >= (score_before * (1.0 - tolerance)):
                        is_valid = True
                else:
                    if score_after >= score_before:
                        is_valid = True

                if is_valid:
                    print(f"[置換成功] '{old_word}' -> '{new_word}' (Score: {score_before:.3f} -> {score_after:.3f})")
                    s_current = s_after_temp
                else:
                    print(f"[置換ブロック] トピック離脱を検知 ('{old_word}' -> '{new_word}')")

        # 5. 退避させていたID数字列を復元
        for idx, pid in enumerate(protected_ids):
            s_current = s_current.replace(f"__PROTECTED_ID_{idx}__", pid)

        return s_current


class NgramTokenizer:
    def __init__(self):
        self.char_counts = Counter()
        self.bigram_counts = Counter()
        self.forced_words = set()
        
    def train(self, corpus_list):
        for sentence in corpus_list:
            if not sentence:
                continue
            self.char_counts.update(sentence)
            bigrams = [sentence[i:i+2] for i in range(len(sentence)-1)]
            self.bigram_counts.update(bigrams)

    def register_words(self, words_list):
        self.forced_words.update(words_list)

    def _get_cohesion_score(self, c1, c2):
        bigram = c1 + c2
        if self.char_counts[c1] == 0 or self.bigram_counts[bigram] == 0:
            return 0.0
        return self.bigram_counts[bigram] / self.char_counts[c1]

    def tokenize(self, text, drop_threshold=0.3):
        if len(text) <= 1:
            return [text]
            
        words = []
        current_word = text[0]
        
        for i in range(len(text) - 1):
            c1, c2 = text[i], text[i+1]
            candidate = current_word + c2
            
            if any(fw.startswith(candidate) for fw in self.forced_words):
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
        with open(file_path, 'wb') as f:
            pickle.dump({'char_counts': self.char_counts, 'bigram_counts': self.bigram_counts}, f)

    def load(self, file_path):
        with open(file_path, 'rb') as f:
            loaded_self = pickle.load(f)
        self.char_counts = loaded_self['char_counts']
        self.bigram_counts = loaded_self['bigram_counts']
        print(f"🚀 モデルを正常に読み込みました: {file_path}")


class Himawaria:
    def __init__(self, directory, maximum_word_replacer_memory=128, min_similarity_threshold=0.45):
        self.direc = directory
        self.maximum_word_replacer_memory = maximum_word_replacer_memory
        self.min_similarity_threshold = min_similarity_threshold

        self.heart = None
        self.last_bot_response = ""
        self.last_bot_base = ""
        self.last_input_content = ""
        self.last_input_similar = ""
        self.last_bot_baseUser = ""
        self.last_input_user = ""
        self.pre_heart = 0
        self.last_user = "あんた"
        self.last_user_bot_replied = "あんた"
        self.current_voice = None
        self.user_log = [None] * 10
        self.rate = 1.0

        self._load_memory_and_settings()

        self.word_replacer_memory_after = self.memory.setdefault("word_replacer_memory_after", [])
        self.word_replacer_memory_before = self.memory.setdefault("word_replacer_memory_before", [])
        self.word_replacer = BM25WordReplacer()

        self.tokenizer = NgramTokenizer()
        try:
            self.tokenizer.load(f"{self.direc}/tokenizer.model")
        except Exception:
            for sen in self.memory["sentence"]:
                self.tokenizer.train([sen[0]])
                self.tokenizer.train([sen[1]])
                
        self.heart = random.randint(0, max(0, len(self.memory["sentence"]) - 1))

    def _save_json(self, file_name, data):
        with open(f"{self.direc}/{file_name}", "w", encoding="utf8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4, sort_keys=True, separators=(',', ': '))

    def _load_json(self, file_name):
        with open(f"{self.direc}/{file_name}", "r", encoding="utf8") as f:
            return json.load(f)

    def _load_memory_and_settings(self):
        try:
            self.memory = self._load_json("memory.json")
            self.settings = self._load_json("settings.json")
        except Exception:
            self.memory = self._load_json("memory_backup.json")
            self.settings = self._load_json("settings.json")
            self._save_json("memory.json", self.memory)
            
        self._save_json("memory_backup.json", self.memory)

    def _add_to_replacer_history(self, after_val, before_val):
        """置換履歴の管理（コマンド文やエラー文を除外）"""
        if after_val is None or before_val is None:
            return

        # ガード3: コマンド文言やエラーメッセージは単語置換用の履歴史料に含めない
        ignore_keywords = ["!command", "エラー:", "権限がありません", "チャンネルがNone"]
        if any(kw in str(after_val) for kw in ignore_keywords) or any(kw in str(before_val) for kw in ignore_keywords):
            return

        self.word_replacer_memory_after.append(after_val)
        self.word_replacer_memory_before.append(before_val)

        limit = self.maximum_word_replacer_memory * (len(self.settings["mynames"].split("|")) + 1)
        self.word_replacer_memory_after = self.word_replacer_memory_after[-limit:]
        self.word_replacer_memory_before = self.word_replacer_memory_before[-limit:]

        self.memory["word_replacer_memory_after"] = self.word_replacer_memory_after
        self.memory["word_replacer_memory_before"] = self.word_replacer_memory_before

    def _apply_word_replacement(self, candidate_text):
        if not candidate_text:
            return candidate_text

        # ガード1: コマンド文字列はそのまま返す
        if candidate_text.strip().startswith("!"):
            return candidate_text

        if hasattr(self, "bm25"):
            self.word_replacer.bm25 = self.bm25

        return self.word_replacer.replace_and_validate(
            response_candidate=candidate_text,
            memory_before_list=self.word_replacer_memory_before,
            memory_after_list=self.word_replacer_memory_after,
            tolerance=0.15
        )

    def learnSentence(self, x, u, save=True, directLearning=False):
        mynames = self.settings["mynames"].split("|")
        if u not in mynames and directLearning and u not in ["!input", "!output", "!system"]:
            u = f"!input-{u}"

        target_u = "!output" if u in mynames else u
        self.memory["sentence"].append([x, target_u])
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
        self._save_json("memory.json", self.memory)
        self.tokenizer.save(f"{self.direc}/tokenizer.model")

    def evalute(self):
        last_sen = self.memory["sentence"][-1][0]
        if last_sen in ["!bad", "!good"]:
            return

        if self.heart + 1 < len(self.memory["sentence"]) - 1:
            next_sen = self.memory["sentence"][self.heart + 1][0]
            if next_sen == "!good":
                print("このメッセージは良い")
                self.learnSentence("!good", "!system")
            elif next_sen == "!bad":
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

    def isAvailable(self, d, b, avail_type=1):
        if b + 2 >= len(self.memory["sentence"]) - 1:
            return avail_type == 1

        next_val = self.memory["sentence"][b + 2][0]
        if next_val == "!bad":
            return False
        if avail_type == 0:
            return (next_val == "!good") and (d >= 0.6)

        return True

    def update_bm25_index(self):
        sentences = self.memory["sentence"]
        corpus = []
        mynames = set(self.settings.get("mynames", "").split("|")) | {"!output"}

        for i, (curr_text, speaker) in enumerate(sentences):
            curr_tag = "[Bot]" if speaker in mynames else f"[{speaker}]"
            curr_doc = f"{curr_tag}{curr_text}"
            
            prev_context = ""
            if i > 0 and "!system" not in sentences[i-1][1]:
                prev_text, prev_speaker = sentences[i-1]
                prev_tag = "[Bot]" if prev_speaker in mynames else f"[{prev_speaker}]"
                prev_context = f"{prev_tag}{prev_text}"
            
            combined_doc = f"{prev_context} {curr_doc} {curr_doc}" if prev_context else f"{curr_doc} {curr_doc}"
            corpus.append(combined_doc)

        self.bm25 = BM25Searcher(k1=1.5, b=0.75)
        self.bm25.fit(corpus)

    def _is_duplicate(self, reply, last_input, last_bot_resp, last_bot_base):
        if len(reply) <= 2:
            return reply == last_bot_resp or reply == last_bot_base
        
        for check_target in [last_input, last_bot_resp, last_bot_base]:
            if Levenshtein.normalized_similarity(reply, check_target) >= 0.85:
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

    def looking(self, x, u, reply=True, force=False):
        print(f"思考中: {x}")
        total_len = len(self.memory["sentence"])
        if total_len < 2:
            return None

        # インデックスの同期チェック
        if not hasattr(self, "bm25") or self.bm25.corpus_size != total_len:
            self.update_bm25_index()

        user_tag = f"[{u}]"
        
        # 1. テキスト本体のみのクエリ
        tokens_text_only = BM25Searcher.tokenize(x)
        
        # 2. 文脈＋タグも含めた評価用クエリ
        context_prefix = f"[Bot]{self.last_bot_response} " if self.last_bot_response else ""
        tokens_context = BM25Searcher.tokenize(f"{context_prefix}{user_tag}{x}")

        # -------------------------------------------------------------
        # 第1段階：テキスト本体（`x`）の一致度で候補上位N件を収集
        # -------------------------------------------------------------
        candidates = []  # (text_score, idx) のリスト

        for idx in range(total_len - 1):
            next_reply = self.memory["sentence"][idx + 1]
            
            # 重複チェックやシステムログの除外
            is_dup = self._is_duplicate(
                next_reply[0], 
                self.last_input_content, 
                self.last_bot_response, 
                self.last_bot_base
            )
            
            if (not is_dup and 
                "!system" not in next_reply[1] and 
                next_reply[0] not in ["!bad", "!good"] and 
                next_reply[1] != "!"):

                # テキスト単体の純粋なBM25スコアを計算
                text_score = self.bm25.get_score(tokens_text_only, idx)
                
                # !good が付いている場合は第1段階でも選考に残りやすくするためスコア補正（1.3倍）
                if self._has_good_tag(idx):
                    text_score *= 1.3

                if text_score > 0.01:
                    candidates.append((text_score, idx))

        if not candidates and force:
            for idx in range(total_len - 1):
                candidates.append((0.0, idx))

        if not candidates:
            return None

        # テキストスコアが高い順にソートし、上位15件に絞り込む
        candidates.sort(key=lambda item: item[0], reverse=True)
        top_candidates = candidates[:15]

        # -------------------------------------------------------------
        # 第2段階：文脈・タグ（`tokens_context`）と !good 優遇を加えて決定
        # -------------------------------------------------------------
        best_b = None
        best_final_score = -1.0

        for text_score, idx in top_candidates:
            context_score = self.bm25.get_score(tokens_context, idx)
            
            # テキストスコア + 文脈補正スコア
            final_score = text_score + (context_score * 0.5)

            # !good が付いている記憶に最終ボーナスを加算（1.5倍掛け）
            if self._has_good_tag(idx):
                final_score *= 1.5

            if final_score > best_final_score:
                best_final_score = final_score
                best_b = idx

        # -------------------------------------------------------------
        # 判定結果の適用
        # -------------------------------------------------------------
        if best_b is not None:
            is_good_mark = " [★good優遇]" if self._has_good_tag(best_b) else ""
            print(f"類似: {self.memory['sentence'][best_b][0]}, idx: {best_b}, FinalScore: {best_final_score:.3f}{is_good_mark}")
            print(f"返信: {self.memory['sentence'][best_b+1][0]}, idx: {best_b+1}")
            self.last_input_similar = self.memory["sentence"][best_b][0]
            self.last_input_user = self.memory["sentence"][best_b][1]
            self.heart = best_b + 1
            self.last_bot_baseUser = self.memory["sentence"][best_b+1][1]
            self.last_bot_base = self.memory["sentence"][best_b+1][0]
            return self.memory["sentence"][best_b+1][0]

        return None

    def generate_response(self, user_input, user_name):
        response_base = self.looking(user_input, user_name)
        if not response_base:
            return "..."
        return self._apply_word_replacement(response_base)

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
            self._add_to_replacer_history(result, self.last_bot_base)
            if self.last_bot_baseUser in ["!", "!output"]:
                for myname in reversed(self.settings["mynames"].split("|")):
                    self._add_to_replacer_history(myname, self.last_bot_baseUser)

        self.last_bot_response = result
        return result

    def nextSpeak(self, is_active_learning=True):
        if not self.isNextOk():
            return None

        self.heart += 1
        result = self.memory["sentence"][self.heart][0]
        self.last_bot_base = result
        self.last_bot_baseUser = self.memory["sentence"][self.heart][1]
        
        if "!" not in self.last_user:
            self.last_userReplied = self.last_user

        self.user_log.append("!")
        self.user_log.pop(0)

        if result is not None:
            result = self._apply_word_replacement(result)
            self._add_to_replacer_history(result, self.last_bot_base)
            for myname in reversed(self.settings["mynames"].split("|")):
                self._add_to_replacer_history(myname, self.last_bot_baseUser)

        self.last_bot_response = result
        return result

    def receive(self, x, u, is_active_learning=True, reply=True, force=False):
        if x is None or u is None:
            return None
        
        self.pre_heart = self.heart
        self.last_input_content = x
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

        result = self.looking(x, u, force=force, reply=reply)
        if result is None:
            self.current_voice = None
            return None

        self._add_to_replacer_history(x, self.last_input_similar)

        if "!system" not in u:
            if u not in ["!", "!output"] and self.last_input_user not in ["!", "!output"]:
                self._add_to_replacer_history(u, self.last_input_user)
            elif self.last_input_user in ["!", "!output"] and u not in ["!", "!output"]:
                for myname in reversed(self.settings["mynames"].split("|")):
                    self._add_to_replacer_history(u, myname)
            elif u in ["!", "!output"] and self.last_input_user not in ["!", "!output"]:
                for myname in reversed(self.settings["mynames"].split("|")):
                    self._add_to_replacer_history(myname, self.last_input_user)

        result = self._apply_word_replacement(result)

        self.current_voice = result
        print(f"座標: {self.heart}")
        print(f"ログ: {self.user_log}")
        print(f"心の声: {result}")
        return result

    def get_settings(self): return self.settings
    def get_memory(self): return self.memory
    def get_current_voice(self): return self.current_voice
    def get_last_bot_response(self): return self.last_bot_response
    def get_last_user(self): return self.last_user


if __name__ == "__main__":
    a = Himawaria("kasen")
    print(a.receive("こんにちは", "ユーザー", True))
    print(a.receive("調子はどう？", "ユーザー", True))