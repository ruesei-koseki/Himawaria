import json
import random
import pickle
from rapidfuzz.distance import Levenshtein
from collections import Counter
import difflib

import math
from collections import Counter

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
        """文字N-gramトークナイザ（分かち書き不要・純Python）"""
        if len(text) < n:
            return [text] if text else []
        return [text[i:i+n] for i in range(len(text) - n + 1)]

    def fit(self, corpus):
        """記憶データ全体からBM25インデックスを事前構築"""
        self.corpus_size = len(corpus)
        if self.corpus_size == 0:
            return

        self.docs_tokens = [self.tokenize(doc) for doc in corpus]
        self.doc_len = [len(tokens) for tokens in self.docs_tokens]
        self.avgdl = sum(self.doc_len) / self.corpus_size if self.corpus_size > 0 else 0

        # ドキュメント頻度 (df) の集計
        df = Counter()
        for tokens in self.docs_tokens:
            frequencies = Counter(tokens)
            self.doc_freqs.append(frequencies)
            for token in frequencies.keys():
                df[token] += 1

        # Okapi BM25 の IDF 計算
        for token, freq in df.items():
            self.idf[token] = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0)

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

class BM25WordReplacer:
    """直近履歴 (before/after) と BM25 を活用した安全な単語置換エンジン"""
    def __init__(self, bm25_searcher=None):
        self.bm25 = bm25_searcher or BM25Searcher()

    def extract_replacement_pairs(self, before_text, after_text, min_len=1):
        """
        before(元文) と after(入力/変換後文) の差分から置換ペア (old, new) を抽出
        例: ("今日は晴れです", "今日は雨です") -> [("晴れ", "雨")]
        """
        matcher = difflib.SequenceMatcher(None, before_text, after_text)
        pairs = []
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'replace':
                old_word = before_text[i1:i2]
                new_word = after_text[j1:j2]
                if len(old_word) >= min_len and len(new_word) >= min_len:
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
                num = tf * (self.bm25.k1 + 1)
                den = tf + self.bm25.k1 * (1 - self.bm25.b + self.bm25.b * (doc_len / avgdl))
                score += idf * (num / den)

        return score

    def replace_and_validate(self, response_candidate, memory_before_list, memory_after_list, tolerance=0.15):
        """
        置換の適用とBM25履歴差分検証を行うメインルーチン
        
        response_candidate : BM25等で引いてきた返答候補文 (beforeベース)
        memory_before_list : 直近最大128件の before 履歴
        memory_after_list  : 直近最大128件の after 履歴
        tolerance          : 文脈親和性スコア低下の許容比率 (0.15 = 15%低下まで許容)
        """
        if not memory_before_list or not memory_after_list:
            return response_candidate

        # 1. 直近履歴から置換可能なペアを全抽出
        candidate_pairs = []
        for b_text, a_text in zip(memory_before_list, memory_after_list):
            pairs = self.extract_replacement_pairs(b_text, a_text)
            candidate_pairs.extend(pairs)

        if not candidate_pairs:
            return response_candidate

        # 2. 直近の文脈トークン集合を作成 (after履歴から構築)
        recent_context_text = " ".join(memory_after_list)
        context_tokens = self.bm25.tokenize(recent_context_text)

        # 3. 応答候補文の中に置換可能な語があるか検索
        s_current = response_candidate
        for old_word, new_word in candidate_pairs:
            if old_word in s_current and old_word != new_word:
                # 仮置換文を生成
                s_after_temp = s_current.replace(old_word, new_word, 1)

                # 置換前後の文脈親和性（BM25スコア）を評価
                score_before = self._calc_text_context_score(s_current, context_tokens)
                score_after = self._calc_text_context_score(s_after_temp, context_tokens)

                # 検証: 置換によってスコアが著しく暴落していないかチェック
                # ( score_before が 0 の場合は直近文脈に含まれる語への置換なら昇格 )
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

        return s_current

class NgramTokenizer:
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
    def __init__(self, directory, maximum_word_replacer_memory=128, min_similarity_threshold=0.45):
        self.direc = directory
        self.maximum_word_replacer_memory = maximum_word_replacer_memory
        # ハルシネーション（無理な引き当て）防止用の最低類似度スコア
        self.min_similarity_threshold = min_similarity_threshold

        self.memory = None
        self.settings = None
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
        self.word_replacer_memory_after = []
        self.word_replacer_memory_before = []
        self.rate = 1.0

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

    def replaceWords(self, x, inputs, inputsHeart):
        """
        単語置換処理: 助詞や短いトークンの破壊的・誤認置換を抑制し、安全な置換を実施
        """
        replacements = []
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
                        if old and new and len(old) >= 2:  # 1文字の助詞などの置換暴走を防止
                            replacements.append((old, new))
                        old, new = "", ""
            if (old or new) and len(old) >= 2:
                replacements.append((old, new))

        # 置換処理 (トークン単位での高精度類似度判定のみ実行)
        res_tokens = list(w3)
        for idx, token in enumerate(res_tokens):
            if len(token) <= 1:
                continue  # 1文字トークン（助詞・ひらがな1文字等）は置換対象から外す
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

    def update_bm25_index(self):
        """記憶データの入力文一覧を取り出してBM25を構築"""
        corpus = [s[0] for s in self.memory["sentence"]]
        self.bm25 = BM25Searcher(k1=1.5, b=0.75)
        self.bm25.fit(corpus)

    def _find_best_match_bm25(self, x, start_idx, end_idx, avail_type, min_score=0.1, strict_speaker=True):
        """BM25を用いた類似記憶検索"""
        if not hasattr(self, "bm25") or self.bm25.corpus_size != len(self.memory["sentence"]):
            self.update_bm25_index()

        query_tokens = BM25Searcher.tokenize(x)
        best_score = min_score
        best_b = None

        for idx in range(start_idx, end_idx):
            if idx + 1 >= len(self.memory["sentence"]):
                break

            next_reply = self.memory["sentence"][idx + 1]

            # BM25スコアの計算
            score = self.bm25.get_score(query_tokens, idx)

            if score > best_score:
                speaker_check = (next_reply[1] != self.memory["sentence"][idx][1]) if strict_speaker else True

                is_dup = self._is_duplicate(
                    next_reply[0], 
                    self.last_input_content, 
                    self.last_bot_response, 
                    self.last_bot_base
                )

                if (speaker_check and 
                    not is_dup and
                    "!system" not in next_reply[1] and
                    next_reply[0] not in ["!bad", "!good"] and
                    next_reply[1] != "!"):
                    
                    if self.isAvailable(score, idx, avail_type):
                        best_score = score
                        best_b = idx

        return best_b, best_score
    
    def _is_duplicate(self, reply, last_input, last_bot_resp, last_bot_base):
        """短文や記号（「？」など）が重複判定で弾かれるのを防ぐヘルパー"""
        # 1〜2文字の超短文・記号の場合
        if len(reply) <= 2:
            # BOTが直前に言ったセリフと完全一致する場合のみ連投防止で弾く
            return reply == last_bot_resp or reply == last_bot_base
        
        # 通常の文章の場合は 0.85 以上の高類似度重複を弾く
        if Levenshtein.normalized_similarity(reply, last_input) >= 0.85:
            return True
        if Levenshtein.normalized_similarity(reply, last_bot_resp) >= 0.85:
            return True
        if Levenshtein.normalized_similarity(reply, last_bot_base) >= 0.85:
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

        f = min(self.heart + 1, total_len - 1)
        prev_f = max(0, self.heart - 1)

        # -------------------------------------------------------------
        # 1. 通常検索 (BM25: strict_speaker=True, 閾値=0.5)
        # -------------------------------------------------------------
        b, d = self._find_best_match_bm25(x, f, total_len - 1, avail_type=0, min_score=0.5, strict_speaker=True)
        if b is None:
            b, d = self._find_best_match_bm25(x, 0, prev_f, avail_type=0, min_score=0.5, strict_speaker=True)

        if b is None:
            b, d = self._find_best_match_bm25(x, f, total_len - 1, avail_type=1, min_score=0.5, strict_speaker=True)
        if b is None:
            b, d = self._find_best_match_bm25(x, 0, prev_f, avail_type=1, min_score=0.5, strict_speaker=True)

        # -------------------------------------------------------------
        # 2. 緩和検索 (BM25: strict_speaker=False ＋ 閾値 0.05 に緩和)
        # -------------------------------------------------------------
        if b is None:
            print("通常マッチなし: 発言者制限解除＋BM25低閾値で再探索します")
            fallback_score = 0.05
            
            b, d = self._find_best_match_bm25(x, f, total_len - 1, avail_type=1, min_score=fallback_score, strict_speaker=False)
            if b is None:
                b, d = self._find_best_match_bm25(x, 0, prev_f, avail_type=1, min_score=fallback_score, strict_speaker=False)

        # -------------------------------------------------------------
        # 3. 救済検索 (完全未学習テキスト対策: 閾値 0.0 で最大スコア記憶を強制抽出)
        # -------------------------------------------------------------
        if b is None and force:
            print("完全ヒットなし: BM25最良の記憶を抽出します")
            b, d = self._find_best_match_bm25(x, 0, total_len - 1, avail_type=1, min_score=0.0, strict_speaker=False)

        # -------------------------------------------------------------
        # 結果の適用
        # -------------------------------------------------------------
        if b is not None:
            print(f"類似: {self.memory['sentence'][b][0]}, idx: {b}, BM25Score: {d:.3f}")
            print(f"返信: {self.memory['sentence'][b+1][0]}, idx: {b+1}")
            self.last_input_similar = self.memory["sentence"][b][0]
            self.last_input_user = self.memory["sentence"][b][1]
            self.heart = b + 1
            self.last_bot_baseUser = self.memory["sentence"][b+1][1]
            self.last_bot_base = self.memory["sentence"][b+1][0]
            return self.memory["sentence"][b+1][0]

        return None

    def generate_response(self, user_input, user_name):
        # 1. 既存の BM25 による応答文抽出 (beforeベースの決定)
        response_base = self.looking(user_input, user_name)

        if not response_base:
            return "..."

        # 2. BM25 インデックスが更新されている場合は検索器を同期
        if hasattr(self, "bm25"):
            self.word_replacer.bm25 = self.bm25

        # 3. 置換と差分検証を実行（安全な応答文を取得）
        final_response = self.word_replacer.replace_and_validate(
            response_candidate=response_base,
            memory_before_list=self.word_replacer_memory_before,
            memory_after_list=self.word_replacer_memory_after,
            tolerance=0.15
        )

        return final_response

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
            if self.last_bot_baseUser in ["!", "!output"]:
                for myname in reversed(self.settings["mynames"].split("|")):
                    self.word_replacer_memory_after.append(myname)
                    self.word_replacer_memory_before.append(self.last_bot_baseUser)
            if len(self.word_replacer_memory_after) > self.maximum_word_replacer_memory:
                self.word_replacer_memory_after = self.word_replacer_memory_after[-self.maximum_word_replacer_memory:]
            if len(self.word_replacer_memory_before) > self.maximum_word_replacer_memory:
                self.word_replacer_memory_before = self.word_replacer_memory_before[-self.maximum_word_replacer_memory:]
            self.memory["word_replacer_memory_after"] = self.word_replacer_memory_after
            self.memory["word_replacer_memory_before"] = self.word_replacer_memory_before

        self.last_bot_response = result
        return result

    def nextSpeak(self, is_active_learning=True):
        if self.isNextOk():
            self.heart += 1
            result = self.memory["sentence"][self.heart][0]
            self.last_bot_base = result
            self.last_bot_baseUser = self.memory["sentence"][self.heart][1]
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
                    self.word_replacer_memory_before.append(self.last_bot_baseUser)
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

        self.word_replacer_memory_after.append(x)
        self.word_replacer_memory_before.append(self.last_input_similar)

        if "!system" not in u:
            if u not in ["!", "!output"] and self.last_input_user not in ["!", "!output"]:
                self.word_replacer_memory_after.append(u)
                self.word_replacer_memory_before.append(self.last_input_user)
            elif self.last_input_user in ["!", "!output"] and u not in ["!", "!output"]:
                for myname in reversed(self.settings["mynames"].split("|")):
                    self.word_replacer_memory_after.append(u)
                    self.word_replacer_memory_before.append(myname)
            elif u in ["!", "!output"] and self.last_input_user not in ["!", "!output"]:
                for myname in reversed(self.settings["mynames"].split("|")):
                    self.word_replacer_memory_after.append(myname)
                    self.word_replacer_memory_before.append(self.last_input_user)
        
        limit = self.maximum_word_replacer_memory * (len(self.settings["mynames"].split("|")) + 1)
        self.word_replacer_memory_after = self.word_replacer_memory_after[-limit:]
        self.word_replacer_memory_before = self.word_replacer_memory_before[-limit:]

        self.memory["word_replacer_memory_after"] = self.word_replacer_memory_after
        self.memory["word_replacer_memory_before"] = self.word_replacer_memory_before
            
        result = self.replaceWords(result, self.word_replacer_memory_after, self.word_replacer_memory_before)
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
    print(a.receive("こんにちは", "小関琉聖だった人", True))
    print(a.receive("調子はどう？", "小関琉聖だった人", True))