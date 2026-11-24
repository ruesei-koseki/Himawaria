import json
import math
import random
import pickle
from rapidfuzz.distance import Levenshtein
from collections import Counter
import difflib


class NpmiTokenizer:
    """
    NPMI（正規化相互情報量）で隣接2文字の結びつきを測り、
    結びつきが弱い箇所で区切るトークナイザ。

    NPMI(x, y) = log( p(x,y) / (p(x) p(y)) ) / -log p(x,y)
      +1 : 常にセットで出現 / 0 : 独立 / -1 : 一緒に出現しない
    """

    def __init__(self):
        self.char_counts = Counter()
        self.bigram_counts = Counter()
        self.total_chars = 0
        self.total_bigrams = 0
        self.forced_words = set()  # 絶対に分割しない単語リスト

    def train(self, corpus_list):
        """1文字と2文字の出現頻度をカウント"""
        for sentence in corpus_list:
            if not sentence:
                continue
            self.char_counts.update(sentence)
            self.total_chars += len(sentence)
            bigrams = [sentence[i:i + 2] for i in range(len(sentence) - 1)]
            self.bigram_counts.update(bigrams)
            self.total_bigrams += len(bigrams)

    def register_words(self, words_list):
        """分割してほしくない単語を辞書として登録する"""
        self.forced_words.update(words_list)

    def npmi(self, c1, c2):
        """隣接2文字のNPMIを返す（-1.0〜1.0）"""
        n12 = self.bigram_counts[c1 + c2]
        n1 = self.char_counts[c1]
        n2 = self.char_counts[c2]
        if n12 == 0 or n1 == 0 or n2 == 0 or self.total_bigrams == 0 or self.total_chars == 0:
            return -1.0
        p12 = n12 / self.total_bigrams
        p1 = n1 / self.total_chars
        p2 = n2 / self.total_chars
        denom = -math.log(p12)
        if denom <= 0:
            # p12 == 1: コーパス中のbigramがこれだけ → 完全に結合
            return 1.0
        return max(-1.0, min(1.0, math.log(p12 / (p1 * p2)) / denom))

    def tokenize(self, text, drop_threshold=0.3):
        """NPMIがしきい値より低い位置で区切る（登録単語は優先して結合）"""
        if not text:
            return []
        if len(text) <= 1:
            return [text]

        words = []
        current_word = text[0]

        for i in range(len(text) - 1):
            c1, c2 = text[i], text[i + 1]
            candidate = current_word + c2

            # 組み立て中の文字列が登録単語の先頭に一致するなら強制結合
            if any(fw.startswith(candidate) for fw in self.forced_words):
                current_word = candidate
            elif self.npmi(c1, c2) < drop_threshold:
                words.append(current_word)
                current_word = c2
            else:
                current_word += c2

        if current_word:
            words.append(current_word)
        return words

    # --- 保存・読み込み ---
    def saveData(self, file_path):
        """学習済みの統計データをファイルに保存する"""
        with open(file_path, 'wb') as f:
            pickle.dump({
                'char_counts': self.char_counts,
                'bigram_counts': self.bigram_counts,
                'total_chars': self.total_chars,
                'total_bigrams': self.total_bigrams,
            }, f)

    save = saveData  # 旧コードの tokenizer.save() 呼び出し互換

    def load(self, file_path):
        """保存されたファイルから統計データを復元する（旧形式のモデルも読める）"""
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        self.char_counts = data['char_counts']
        self.bigram_counts = data['bigram_counts']
        self.total_chars = data.get('total_chars', sum(self.char_counts.values()))
        self.total_bigrams = data.get('total_bigrams', sum(self.bigram_counts.values()))
        print(f"🚀 モデルを正常に読み込みました: {file_path}")


class BM25LIndex:
    """
    BM25L の転置インデックス（1文書ずつ追加可能）。

    idf   = log((N + 1) / (df + 0.5))
    c_td  = tf / (1 - b + b * |d| / avgdl)
    score = Σ idf * (k1 + 1) * (c_td + δ) / (k1 + c_td + δ)

    BM25 は長い文書のスコアを下げすぎる傾向があり、BM25L は δ を足してそれを補正する。
    """

    def __init__(self, k1=1.5, b=0.75, delta=0.5):
        self.k1 = k1
        self.b = b
        self.delta = delta
        self.clear()

    def clear(self):
        self.postings = {}   # term -> {doc_id: tf}
        self.doc_len = []    # doc_id -> 文書長（term数）
        self.total_len = 0

    def __len__(self):
        return len(self.doc_len)

    def add(self, terms):
        """文書（termのリスト）を追加し、doc_id（=追加順）を返す"""
        doc_id = len(self.doc_len)
        self.doc_len.append(len(terms))
        self.total_len += len(terms)
        for t, tf in Counter(terms).items():
            self.postings.setdefault(t, {})[doc_id] = tf
        return doc_id

    def _idf(self, term):
        df = len(self.postings.get(term, ()))
        return math.log((len(self.doc_len) + 1) / (df + 0.5))

    def _term_score(self, idf, tf, dl, avgdl):
        ctd = tf / (1 - self.b + self.b * dl / avgdl)
        return idf * (self.k1 + 1) * (ctd + self.delta) / (self.k1 + ctd + self.delta)

    def search(self, query_terms):
        """
        returns (scores, self_score)
          scores:     {doc_id: 生スコア}
          self_score: クエリ自身を文書とみなしたときのスコア（正規化用の上限目安）
        """
        n = len(self.doc_len)
        if n == 0 or not query_terms:
            return {}, 0.0
        avgdl = (self.total_len / n) or 1.0
        qcount = Counter(query_terms)

        scores = {}
        self_score = 0.0
        qlen = len(query_terms)
        for t, qtf in qcount.items():
            idf = self._idf(t)
            self_score += self._term_score(idf, qtf, qlen, avgdl)
            posting = self.postings.get(t)
            if not posting:
                continue
            for doc_id, tf in posting.items():
                scores[doc_id] = scores.get(doc_id, 0.0) + self._term_score(
                    idf, tf, self.doc_len[doc_id], avgdl)
        return scores, self_score


class Himawaria:
    # NPMIトークナイザの区切りしきい値（低いほど細かく、高いほど長い単語になる）
    NPMI_THRESHOLD = 0.3
    # 通常の返信候補（type 0）に要求する正規化BM25Lスコア（0〜1）
    MIN_GOOD_SIMILARITY = 0.6

    def __init__(self, directory, maximum_word_replacer_memory=128):
        self.direc = directory
        self.maximum_word_replacer_memory = maximum_word_replacer_memory

        self.memory = None #別途読み込むデータ
        self.settings = None #設定
        self.heart = None #今の気持ち(ログの座標で表される)
        self.last_bot_response = "" #最後のbotの言葉
        self.last_bot_base = "" #最後のbotのベース発言
        self.last_input_content = "" #最後に聞いた言葉
        self.last_input_similar = "" #最後に聞いた言葉
        self.last_bot_baseUser = "" #過去に今からBOTが話すことと似た話をしてたユーザー
        self.last_input_user = "" #過去に今ユーザーから話されたことと似た話をしてたユーザー
        self.pre_heart = 0 #一つ前の気持ち
        self.last_user = "あんた" #最後に話したユーザー
        self.last_user_bot_replied = "あんた" #最後に返信したユーザー
        self.current_voice = None #心の中の声
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

        try:
            self.memory["word_replacer_memory_after"]
        except:
            self.memory["word_replacer_memory_after"] = []
        try:
            self.memory["word_replacer_memory_before"]
        except:
            self.memory["word_replacer_memory_before"] = []

        self.word_replacer_memory_after = self.memory["word_replacer_memory_after"]
        self.word_replacer_memory_before = self.memory["word_replacer_memory_before"]

        self.tokenizer = NpmiTokenizer()
        try:
            self.tokenizer.load(self.direc+"/tokenizer.model")
        except:
            for sen in self.memory["sentence"]:
                self.tokenizer.train([sen[0]])
                self.tokenizer.train([sen[1]])

        # BM25L 検索インデックス（初回の検索時に構築する）
        self.index = BM25LIndex()
        self._index_dirty = True

        self.heart = random.randint(0, len(self.memory["sentence"]) - 1) #今の気持ち(ログの座標で表される)

        self.current_voice = None

    # ---------- 検索インデックス ----------
    def _terms(self, text):
        """BM25L用のterm列: NPMI分かち書き + 文字bigram（短文・表記ゆれに強くする）"""
        if not isinstance(text, str) or not text:
            return []
        words = self.tokenizer.tokenize(text, drop_threshold=self.NPMI_THRESHOLD)
        bigrams = [text[i:i + 2] for i in range(len(text) - 1)]
        return words + bigrams

    def _ensure_index(self):
        """sentence の挿入・削除で doc_id がずれたときはここで作り直す"""
        if not self._index_dirty:
            return
        self.index.clear()
        for sen in self.memory["sentence"]:
            self.index.add(self._terms(sen[0]))
        self._index_dirty = False

    def learnSentence(self, x, u, save=True, directLearning=False):
        #directLearningがTrueのときに自分の名前以外を無効にする
        if u not in self.settings["mynames"].split("|") and directLearning and u not in ["!input", "!output", "!system"]:
            u = "!input-"+u

        #言葉を脳に記録する
        if u in self.settings["mynames"].split("|"):
            self.memory["sentence"].append([x, "!output"])
        else:
            self.memory["sentence"].append([x, u])
        self.tokenizer.train([x, u])
        if not self._index_dirty:
            self.index.add(self._terms(x))

        if len(self.memory["sentence"]) >= 1600000:
            while len(self.memory["sentence"]) >= 1600000:
                del self.memory["sentence"][0]
                if self.heart >= 5 or self.pre_heart >= 5:
                    self.heart -= 1
                    self.pre_heart -= 1
            self._index_dirty = True  # 先頭を削ったので doc_id がずれる
        if save:
            self.save()

    def save(self):
        with open(self.direc+"/memory.json", "w", encoding="utf8") as f:
            json.dump(self.memory, f, ensure_ascii=False, indent=4, sort_keys=True, separators=(',', ': '))
        self.tokenizer.saveData(self.direc+"/tokenizer.model")

    saveData = save  # 旧コードの self.saveData() 呼び出し互換

    def evalute(self):
        flag1 = False
        flag2 = False
        if self.memory["sentence"][len(self.memory["sentence"])-1][0] != "!bad" and self.memory["sentence"][len(self.memory["sentence"])-1][0] != "!good":
            if self.heart+1 < len(self.memory["sentence"]) - 1:
                if self.memory["sentence"][self.heart+1][0] == "!good" and self.memory["sentence"][-1][0] != "!good":
                    flag1 = True
            if flag1:
                print("このメッセージは良い")
                self.learnSentence("!good", "!system")

        if self.memory["sentence"][len(self.memory["sentence"])-1][0] != "!bad" and self.memory["sentence"][len(self.memory["sentence"])-1][0] != "!good" and not flag1:
            if self.heart+1 < len(self.memory["sentence"]) - 1:
                if self.memory["sentence"][self.heart+1][0] == "!bad" and self.memory["sentence"][-1][0] != "!bad":
                    flag2 = True
            if flag2:
                print("このメッセージは悪い")
                self.learnSentence("!bad", "!system")

    def isNextOk(self):
        if len(self.memory["sentence"]) - 1 <= self.heart+1:
            return False
        else:
            return self.last_input_content != self.memory["sentence"][self.heart+1][0] and self.last_bot_response != self.memory["sentence"][self.heart+1][0] and self.memory["sentence"][self.heart+1][1] == self.memory["sentence"][self.heart][1] and self.memory["sentence"][self.heart+1][1] != "!" and "!system" not in self.memory["sentence"][self.heart+1][1]

    def replaceWords(self, x, inputs, inputsHeart):
        replacements = []
        self.tokenizer.train([x])
        w3 = self.tokenizer.tokenize(x, drop_threshold=self.NPMI_THRESHOLD)
        for i in range(0, len(inputs)):
            if not inputs[i] or not inputsHeart[i]:
                continue
            self.tokenizer.train([inputs[i]])
            self.tokenizer.train([inputsHeart[i]])
            w1 = self.tokenizer.tokenize(inputs[i], drop_threshold=self.NPMI_THRESHOLD)
            w2 = self.tokenizer.tokenize(inputsHeart[i], drop_threshold=self.NPMI_THRESHOLD)

            # 差分を取得
            diffs = list(difflib.ndiff(w2, w1))
            old = ""
            new = ""
            for diff in diffs:
                tag = diff[:2]
                content = diff[2:]

                if tag == "- ":
                    old += content
                elif tag == "+ ":
                    new += content
                elif tag == "  ":
                    if old or new:
                        replacements.append((old, new))
                        old = ""
                        new = ""
                    replacements.append((content, content))
            if old or new:
                replacements.append((old, new))

        # 置換処理
        i = 0
        temp = ""
        temp2 = ""
        temp3 = []
        for wo3 in w3:
            for old, new in reversed(replacements):
                if Levenshtein.normalized_similarity(wo3, old) >= 0.85:
                    print("{} => {}".format(old, new))
                    w3[i] = new
                    temp = ""
                    temp2 = ""
                    temp3 = []
                    break
                elif Levenshtein.normalized_similarity(wo3, temp) >= 0.85:
                    print("{} => {}".format(temp, temp2))
                    for t3 in temp3:
                        w3[t3] = ""
                    w3[i] = temp2
                    temp = ""
                    temp2 = ""
                    temp3 = []
                    break
                elif wo3 in old:
                    temp = old.replace(wo3, "")
                    temp2 = new
                    temp3.append(i)
                    break
                elif wo3 in temp:
                    temp = temp.replace(wo3, "")
                    temp3.append(i)
                    break
            i += 1
        result = "".join(w3)
        return result


    def isAvailable(self, d, b, type=1):
        if type == 0:
            flag = False
            for i in range(1):
                if b+2+i < len(self.memory["sentence"]) - 1:
                    if self.memory["sentence"][b+2+i][0] == "!good":
                        flag = True
                        break
            for i in range(1):
                if b+2+i < len(self.memory["sentence"]) - 1:
                    if self.memory["sentence"][b+2+i][0] == "!bad":
                        flag = False
                        break
            return flag and d >= self.MIN_GOOD_SIMILARITY
        elif type == 1:
            flag = True
            for i in range(1):
                if b+2+i < len(self.memory["sentence"]) - 1:
                    if self.memory["sentence"][b+2+i][0] == "!bad":
                        flag = False
                        break
            return flag

    # ---------- 類似発言の検索（BM25L） ----------
    def _rank_candidates(self, x):
        """
        BM25Lでxに似た過去発言をスコア順に並べて [(index, 正規化スコア0〜1), ...] を返す。
        探索範囲は従来どおり「heart+1 以降」→「heart-2 以前」で、
        同点の場合は従来の走査順（heart直後が優先）にする。
        """
        self._ensure_index()
        scores, self_score = self.index.search(self._terms(x))
        if not scores or self_score <= 0:
            return []

        n = len(self.memory["sentence"])
        heart = self.heart
        ranked = []
        for i, s in scores.items():
            if heart + 1 <= i <= n - 2:
                order = i - heart
            elif 0 <= i <= heart - 2:
                order = n + i
            else:
                continue
            ranked.append((-s, order, i, min(1.0, s / self_score)))
        ranked.sort()
        return [(i, c) for _, _, i, c in ranked]

    def _candidate_ok(self, i, strict):
        """i番目の発言への返信(i+1番目)として使ってよいか"""
        sentences = self.memory["sentence"]
        nxt = sentences[i+1]
        if nxt[0] == "!bad" or nxt[0] == "!good":
            return False
        if "!system" in nxt[1] or "!input" in nxt[1]:
            return False
        if strict and (nxt[1] == "!" or nxt[1] == sentences[i][1]):
            return False
        return (Levenshtein.normalized_similarity(nxt[0], self.last_input_content) < 0.85 and
                Levenshtein.normalized_similarity(nxt[0], self.last_bot_response) < 0.85 and
                Levenshtein.normalized_similarity(nxt[0], self.last_bot_base) < 0.85)

    def looking(self, x, u, reply=True, force=False):
        #過去の発言をもとに考える
        print("思考中: {}".format(x))

        ranked = self._rank_candidates(x)

        # (厳しい条件・!goodが続く候補のみ) → (厳しい条件) → (緩い条件) の順に緩めていく
        for strict, avail_type in ((True, 0), (True, 1), (False, 1)):
            for b, d in ranked:
                if not self._candidate_ok(b, strict):
                    continue
                if not self.isAvailable(d, b, avail_type):
                    continue
                sentences = self.memory["sentence"]
                print("類似: {}, {}, {}".format(sentences[b][0], b, d))
                print("返信: {}, {}".format(sentences[b+1][0], b+1))
                self.last_input_similar = sentences[b][0]
                self.last_input_user = sentences[b][1]
                self.heart = b+1
                self.last_bot_baseUser = sentences[b+1][1]
                self.last_bot_base = sentences[b+1][0]
                return sentences[b+1][0]

        return None

    def record(self):
        self.learnSentence(self.current_voice, "!")
        self.evalute()

    def speakFreely(self, is_active_learning=True):
        #自由に話す
        result = self.current_voice
        if "!" not in self.last_user:
            self.last_userReplied = self.last_user
        self.user_log.append("!")
        self.user_log.pop(0)
        if result != None:
            self.word_replacer_memory_after.append(result)
            self.word_replacer_memory_before.append(self.last_bot_base)
            if self.last_bot_baseUser == "!" or self.last_bot_baseUser == "!output":
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
            if result != None:
                result = self.replaceWords(result, self.word_replacer_memory_after, self.word_replacer_memory_before)

                self.word_replacer_memory_after.append(result)
                self.word_replacer_memory_before.append(self.last_bot_base)
                for myname in reversed(self.settings["mynames"].split("|")):
                    self.word_replacer_memory_after.append(myname)
                    self.word_replacer_memory_before.append(self.last_bot_baseUser)
                if len(self.word_replacer_memory_after) > self.maximum_word_replacer_memory*(len(self.settings["mynames"].split("|"))+1):
                    self.word_replacer_memory_after = self.word_replacer_memory_after[-self.maximum_word_replacer_memory*(len(self.settings["mynames"].split("|"))+1):]
                if len(self.word_replacer_memory_before) > self.maximum_word_replacer_memory*(len(self.settings["mynames"].split("|"))+1):
                    self.word_replacer_memory_before = self.word_replacer_memory_before[-self.maximum_word_replacer_memory*(len(self.settings["mynames"].split("|"))+1):]
                self.memory["word_replacer_memory_after"] = self.word_replacer_memory_after
                self.memory["word_replacer_memory_before"] = self.word_replacer_memory_before

            self.last_bot_response = result
            return result
        else:
            return None


    def receive(self, x, u, is_active_learning=True, reply=True, force=False):
        if x == None or u == None: return

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
                self._index_dirty = True  # 途中に挿入したので doc_id がずれる
            if x == "!good" and self.memory["sentence"][self.heart+1][0] != "!good":
                self.memory["sentence"].insert(self.heart+1, ["!good", "!"])
                self._index_dirty = True
        result = self.looking(x, u, force=force, reply=reply)

        if result == None:
            self.current_voice = None
            return

        self.word_replacer_memory_after.append(x)
        self.word_replacer_memory_before.append(self.last_input_similar)

        if "!system" not in u:
            if u != "!" and u != "!output" and self.last_input_user != "!" and self.last_input_user != "!output":
                self.word_replacer_memory_after.append(u)
                self.word_replacer_memory_before.append(self.last_input_user)
            elif (self.last_input_user == "!" or self.last_input_user == "!output") and u != "!" and u != "!output":
                for myname in reversed(self.settings["mynames"].split("|")):
                    self.word_replacer_memory_after.append(u)
                    self.word_replacer_memory_before.append(myname)
            elif (u == "!" or u == "!output") and self.last_input_user != "!" and self.last_input_user != "!output":
                for myname in reversed(self.settings["mynames"].split("|")):
                    self.word_replacer_memory_after.append(myname)
                    self.word_replacer_memory_before.append(self.last_input_user)
        if len(self.word_replacer_memory_after) > self.maximum_word_replacer_memory*(len(self.settings["mynames"].split("|"))+1):
            self.word_replacer_memory_after = self.word_replacer_memory_after[-self.maximum_word_replacer_memory*(len(self.settings["mynames"].split("|"))+1):]
        if len(self.word_replacer_memory_before) > self.maximum_word_replacer_memory*(len(self.settings["mynames"].split("|"))+1):
            self.word_replacer_memory_before = self.word_replacer_memory_before[-self.maximum_word_replacer_memory*(len(self.settings["mynames"].split("|"))+1):]

        self.memory["word_replacer_memory_after"] = self.word_replacer_memory_after
        self.memory["word_replacer_memory_before"] = self.word_replacer_memory_before

        result = self.replaceWords(result, self.word_replacer_memory_after, self.word_replacer_memory_before)
        result = result.replace("[YOU]", self.last_user)
        result = result.replace("[ME]", self.settings["myname"])
        self.current_voice = result
        print("座標: {}".format(self.heart))
        print("ログ: {}".format(self.user_log))
        print("心の声: {}".format(result))

    def get_settings(self):
        return self.settings

    def get_memory(self):
        return self.memory

    def get_current_voice(self):
        return self.current_voice

    def get_last_user(self):
        return self.last_user


if __name__ == "__main__":
    a = Himawaria("kasen")
    print(a.receive("こんにちは", "小関琉聖だった人", True))
    print(a.receive("調子はどう？", "小関琉聖だった人", True))