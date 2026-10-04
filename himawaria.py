import json
import random
import pickle
import math
import re
import difflib
from collections import Counter
from rapidfuzz.distance import Levenshtein

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

    def update_bm25_index(self):
        """Botの発言名 '!' を基準に過去へ動的に遡り、2つ前までの論理文脈を構築（検索・応答選択用）"""
        sentences = self.memory["sentence"]
        corpus = []

        for i in range(len(sentences)):
            curr_text, speaker = sentences[i][0], sentences[i][1]
            curr_tag = "[!]" if speaker == "!" else f"[{speaker}]"
            curr_doc = f"{curr_tag}{curr_text}"

            prev_bot_doc = ""
            prev_input_doc = ""

            j = i - 1
            while j >= 0:
                s_text, s_speaker = sentences[j][0], sentences[j][1]
                s_tag = "[!]" if s_speaker == "!" else f"[{s_speaker}]"

                if not prev_bot_doc and s_speaker == "!":
                    prev_bot_doc = f"{s_tag}{s_text}"
                elif prev_bot_doc and s_speaker != "!":
                    prev_input_doc = f"{s_tag}{s_text}"
                    break
                j -= 1

            contexts = [c for c in [prev_input_doc, prev_bot_doc, curr_doc, curr_doc] if c]
            combined_doc = " ".join(contexts)

            corpus.append(combined_doc)

        self.bm25 = BM25Searcher(k1=1.5, b=0.75)
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

    def looking(self, x, u, reply=True, force=False):
        print(f"思考中: {x}")
        total_len = len(self.memory["sentence"])
        if total_len < 2:
            return None

        if not hasattr(self, "bm25") or self.bm25.corpus_size != total_len:
            self.update_bm25_index()

        user_tag = f"[{u}]"
        
        tokens_text_only = BM25Searcher.tokenize(x)
        
        context_parts = []
        if hasattr(self, "last_input_content") and self.last_input_content:
            prev_user_tag = f"[{self.last_input_user}]" if getattr(self, "last_input_user", None) else "[User]"
            context_parts.append(f"{prev_user_tag}{self.last_input_content}")

        if self.last_bot_response:
            context_parts.append(f"[!]{self.last_bot_response}")

        context_parts.append(f"{user_tag}{x}")
        context_parts.append(f"{x}")

        context_str = " ".join(context_parts)
        tokens_context = BM25Searcher.tokenize(context_str)

        candidates = []

        for idx in range(total_len - 1):
            next_reply = self.memory["sentence"][idx + 1]
            
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

                text_score = self.bm25.get_score(tokens_text_only, idx)

                if self._has_good_tag(idx):
                    text_score *= 1.3

                if text_score > 0.01:
                    candidates.append((text_score, idx))

        if not candidates and force:
            for idx in range(total_len - 1):
                candidates.append((0.0, idx))

        if not candidates:
            return None

        candidates.sort(key=lambda item: item[0], reverse=True)
        top_candidates = candidates[:15]

        best_b = None
        best_final_score = -1.0

        for text_score, idx in top_candidates:
            context_score = self.bm25.get_score(tokens_context, idx)
            raw_context = context_score * 0.3
            capped_context = min(raw_context, text_score)
            final_score = text_score + capped_context

            if final_score > best_final_score:
                best_final_score = final_score
                best_b = idx

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

        # BM25WordReplacerは使わず、通常の置換機能かそのままの候補を返す形に調整
        # 必要であればここで self.replaceWords(...) などを適用可能ですが、
        # 基本の応答生成としてはそのままベースを返します
        return response_base

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
                for myname in self.settings["mynames"].split("|"):
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
                for myname in self.settings["mynames"].split("|"):
                    self.word_replacer_memory_after.append(u)
                    self.word_replacer_memory_before.append(myname)
            elif u in ["!", "!output"] and self.last_input_user not in ["!", "!output"]:
                for myname in self.settings["mynames"].split("|"):
                    self.word_replacer_memory_after.append(myname)
                    self.word_replacer_memory_before.append(self.last_input_user)

        result = self.replaceWords(result, self.word_replacer_memory_after, self.word_replacer_memory_before)

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