import json
import random
import pickle
from rapidfuzz.distance import Levenshtein
from collections import Counter
import difflib

class NgramTokenizer:
    def __init__(self):
        self.char_counts = Counter()
        self.bigram_counts = Counter()
        self.forced_words = set()  # 追加：絶対に分割しない単語リスト
        
    def train(self, corpus_list):
        """1文字と2文字の出現頻度をカウント"""
        for sentence in corpus_list:
            if not sentence:
                continue
            self.char_counts.update(sentence)
            bigrams = [sentence[i:i+2] for i in range(len(sentence)-1)]
            self.bigram_counts.update(bigrams)

    def register_words(self, words_list):
        """
        追加：分割してほしくない単語を辞書として登録する
        例: tokenizer.register_words(["学校", "マック"])
        """
        self.forced_words.update(words_list)

    def _get_cohesion_score(self, c1, c2):
        """2文字の結合度スコア（条件付き確率）"""
        bigram = c1 + c2
        if self.char_counts[c1] == 0 or self.bigram_counts[bigram] == 0:
            return 0.0
        return self.bigram_counts[bigram] / self.char_counts[c1]

    def tokenize(self, text, drop_threshold=0.3):
        """結合度スコアがしきい値より低いタイミングで区切る（登録単語は優先して結合）"""
        if len(text) <= 1:
            return [text]
            
        words = []
        current_word = text[0]
        
        for i in range(len(text) - 1):
            c1, c2 = text[i], text[i+1]
            candidate = current_word + c2
            
            # --- 追加：現在組み立て中の文字列が「登録単語」の先頭に一致するかチェック ---
            is_part_of_forced_word = any(
                fw.startswith(candidate) for fw in self.forced_words
            )
            
            if is_part_of_forced_word:
                # 登録単語の一部なら、しきい値を無視して強制結合
                current_word = candidate
            else:
                # 通常のN-gramしきい値判定
                score = self._get_cohesion_score(c1, c2)
                if score < drop_threshold:
                    words.append(current_word)
                    current_word = c2
                else:
                    current_word += c2
                
        if current_word:
            words.append(current_word)
            
        return words
    
    # --- ここから保存・読み込みの機能を追加 ---
    def saveData(self, file_path):
        """学習済みの統計データをファイルに保存する"""
        # 保存したいデータ（インスタンス変数）を辞書にまとめる
        self_to_save = {
            'char_counts': self.char_counts,
            'bigram_counts': self.bigram_counts
        }
        with open(file_path, 'wb') as f:
            pickle.dump(self_to_save, f)
        #print(f"🎉 モデルを正常に保存しました: {file_path}")

    def load(self, file_path):
        """保存されたファイルから統計データを復元する"""
        with open(file_path, 'rb') as f:
            loaded_self = pickle.load(f)
        # 復元したデータをクラスにセット
        self.char_counts = loaded_self['char_counts']
        self.bigram_counts = loaded_self['bigram_counts']
        print(f"🚀 モデルを正常に読み込みました: {file_path}")

class Himawaria:
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

        self.tokenizer = NgramTokenizer()
        try:
            self.tokenizer.load(self.direc+"/tokenizer.model")
        except:
            for sen in self.memory["sentence"]:
                self.tokenizer.train([sen[0]])
                self.tokenizer.train([sen[1]])
        self.heart = random.randint(0, len(self.memory["sentence"]) - 1) #今の気持ち(ログの座標で表される)

        self.current_voice = None


    def learnSentence(self, x, u, save=True, directLearning=False):
        #if len(self.memory["sentence"]) >= 1000 or directLearning:
        
        #directLearningがTrueのときに自分の名前以外を無効にする
        if u not in self.settings["mynames"].split("|") and directLearning and u not in ["!input", "!output", "!system"]:
            u = "!input-"+u

        #言葉を脳に記録する
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
        w3 = self.tokenizer.tokenize(x, drop_threshold=0.4)
        for i in range(0, len(inputs)):
            if not inputs[i] or not inputsHeart[i]:
                continue
            self.tokenizer.train([inputs[i]])
            self.tokenizer.train([inputsHeart[i]])
            w1 = self.tokenizer.tokenize(inputs[i], drop_threshold=0.4)
            w2 = self.tokenizer.tokenize(inputsHeart[i], drop_threshold=0.4)

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
            return flag and d >= 0.6
        elif type == 1:
            flag = True
            for i in range(1):
                if b+2+i < len(self.memory["sentence"]) - 1:
                    if self.memory["sentence"][b+2+i][0] == "!bad":
                        flag = False
                        break
            return flag

    def looking(self, x, u, reply=True, force=False):
        #過去の発言をもとに考える
        print("思考中: {}".format(x))

        #今の気持ちから考える
        f = self.heart+1
        t = len(self.memory["sentence"]) - 1
        i = f
        d = 0
        b = None
        for sen in self.memory["sentence"][f:t]:
            c = Levenshtein.normalized_similarity(x, sen[0]) 
            if c > d:
                if (i != len(self.memory["sentence"]) and
                    Levenshtein.normalized_similarity(self.memory["sentence"][i+1][0], self.last_input_content) < 0.85 and
                    Levenshtein.normalized_similarity(self.memory["sentence"][i+1][0], self.last_bot_response) < 0.85 and
                    Levenshtein.normalized_similarity(self.memory["sentence"][i+1][0], self.last_bot_base) < 0.85 and
                    "!system" not in self.memory["sentence"][i+1][1] and
                    "!input" not in self.memory["sentence"][i+1][1] and
                    self.memory["sentence"][i+1][0] != "!bad" and
                    self.memory["sentence"][i+1][0] != "!good" and
                    self.memory["sentence"][i+1][1] != "!" and
                    self.memory["sentence"][i+1][1] != self.memory["sentence"][i][1]):
                    if self.isAvailable(c, i, 0):
                        d = c
                        b = i
            i += 1
        f = 0
        t = self.heart-1
        i = f
        for sen in self.memory["sentence"][f:t]:
            c = Levenshtein.normalized_similarity(x, sen[0]) 
            if c > d:
                if (i != len(self.memory["sentence"]) and
                    Levenshtein.normalized_similarity(self.memory["sentence"][i+1][0], self.last_input_content) < 0.85 and
                    Levenshtein.normalized_similarity(self.memory["sentence"][i+1][0], self.last_bot_response) < 0.85 and
                    Levenshtein.normalized_similarity(self.memory["sentence"][i+1][0], self.last_bot_base) < 0.85 and
                    "!system" not in self.memory["sentence"][i+1][1] and
                    "!input" not in self.memory["sentence"][i+1][1] and
                    self.memory["sentence"][i+1][0] != "!bad" and
                    self.memory["sentence"][i+1][0] != "!good" and
                    self.memory["sentence"][i+1][1] != "!" and
                    self.memory["sentence"][i+1][1] != self.memory["sentence"][i][1]):
                    if self.isAvailable(c, i, 0):
                        d = c
                        b = i
            i += 1
        if b != None:
            print("類似: {}, {}, {}".format(self.memory["sentence"][b][0], b, d))
            print("返信: {}, {}".format(self.memory["sentence"][b+1][0], b+1))
            self.last_input_similar = self.memory["sentence"][b][0]
            self.last_input_user = self.memory["sentence"][b][1]
            self.heart = b+1
            self.last_bot_baseUser = self.memory["sentence"][b+1][1]
            self.last_bot_base = self.memory["sentence"][b+1][0]
            return self.memory["sentence"][b+1][0]

        #今の気持ちから考える
        f = self.heart+1
        t = len(self.memory["sentence"]) - 1
        i = f
        d = 0
        b = None
        for sen in self.memory["sentence"][f:t]:
            c = Levenshtein.normalized_similarity(x, sen[0]) 
            if c > d:
                if (i != len(self.memory["sentence"]) and
                    Levenshtein.normalized_similarity(self.memory["sentence"][i+1][0], self.last_input_content) < 0.85 and
                    Levenshtein.normalized_similarity(self.memory["sentence"][i+1][0], self.last_bot_response) < 0.85 and
                    Levenshtein.normalized_similarity(self.memory["sentence"][i+1][0], self.last_bot_base) < 0.85 and
                    "!system" not in self.memory["sentence"][i+1][1] and
                    "!input" not in self.memory["sentence"][i+1][1] and
                    self.memory["sentence"][i+1][0] != "!bad" and
                    self.memory["sentence"][i+1][0] != "!good" and
                    self.memory["sentence"][i+1][1] != "!" and
                    self.memory["sentence"][i+1][1] != self.memory["sentence"][i][1]):
                    if self.isAvailable(c, i, 1):
                        d = c
                        b = i
            i += 1
        f = 0
        t = self.heart-1
        i = f
        for sen in self.memory["sentence"][f:t]:
            c = Levenshtein.normalized_similarity(x, sen[0]) 
            if c > d:
                if (i != len(self.memory["sentence"]) and
                    Levenshtein.normalized_similarity(self.memory["sentence"][i+1][0], self.last_input_content) < 0.85 and
                    Levenshtein.normalized_similarity(self.memory["sentence"][i+1][0], self.last_bot_response) < 0.85 and
                    Levenshtein.normalized_similarity(self.memory["sentence"][i+1][0], self.last_bot_base) < 0.85 and
                    "!system" not in self.memory["sentence"][i+1][1] and
                    "!input" not in self.memory["sentence"][i+1][1] and
                    self.memory["sentence"][i+1][0] != "!bad" and
                    self.memory["sentence"][i+1][0] != "!good" and
                    self.memory["sentence"][i+1][1] != "!" and
                    self.memory["sentence"][i+1][1] != self.memory["sentence"][i][1]):
                    if self.isAvailable(c, i, 1):
                        d = c
                        b = i
            i += 1
        if b != None:
            print("類似: {}, {}, {}".format(self.memory["sentence"][b][0], b, d))
            print("返信: {}, {}".format(self.memory["sentence"][b+1][0], b+1))
            self.last_input_similar = self.memory["sentence"][b][0]
            self.last_input_user = self.memory["sentence"][b][1]
            self.heart = b+1
            self.last_bot_baseUser = self.memory["sentence"][b+1][1]
            self.last_bot_base = self.memory["sentence"][b+1][0]
            return self.memory["sentence"][b+1][0]

        #今の気持ちから考える
        f = self.heart+1
        t = len(self.memory["sentence"]) - 1
        i = f
        d = 0
        b = None
        for sen in self.memory["sentence"][f:t]:
            c = Levenshtein.normalized_similarity(x, sen[0]) 
            if c > d:
                if (i != len(self.memory["sentence"]) and
                    Levenshtein.normalized_similarity(self.memory["sentence"][i+1][0], self.last_input_content) < 0.85 and
                    Levenshtein.normalized_similarity(self.memory["sentence"][i+1][0], self.last_bot_response) < 0.85 and
                    Levenshtein.normalized_similarity(self.memory["sentence"][i+1][0], self.last_bot_base) < 0.85 and
                    "!system" not in self.memory["sentence"][i+1][1] and
                    "!input" not in self.memory["sentence"][i+1][1] and
                    self.memory["sentence"][i+1][0] != "!bad" and
                    self.memory["sentence"][i+1][0] != "!good"):
                    if self.isAvailable(c, i, 1):
                        d = c
                        b = i
            i += 1
        f = 0
        t = self.heart-1
        i = f
        for sen in self.memory["sentence"][f:t]:
            c = Levenshtein.normalized_similarity(x, sen[0]) 
            if c > d:
                if (i != len(self.memory["sentence"]) and
                    Levenshtein.normalized_similarity(self.memory["sentence"][i+1][0], self.last_input_content) < 0.85 and
                    Levenshtein.normalized_similarity(self.memory["sentence"][i+1][0], self.last_bot_response) < 0.85 and
                    Levenshtein.normalized_similarity(self.memory["sentence"][i+1][0], self.last_bot_base) < 0.85 and
                    "!system" not in self.memory["sentence"][i+1][1] and
                    "!input" not in self.memory["sentence"][i+1][1] and
                    self.memory["sentence"][i+1][0] != "!bad" and
                    self.memory["sentence"][i+1][0] != "!good"):
                    if self.isAvailable(c, i, 1):
                        d = c
                        b = i
            i += 1
        if b != None:
            print("類似: {}, {}, {}".format(self.memory["sentence"][b][0], b, d))
            print("返信: {}, {}".format(self.memory["sentence"][b+1][0], b+1))
            self.last_input_similar = self.memory["sentence"][b][0]
            self.last_input_user = self.memory["sentence"][b][1]
            self.heart = b+1
            self.last_bot_baseUser = self.memory["sentence"][b+1][1]
            self.last_bot_base = self.memory["sentence"][b+1][0]
            return self.memory["sentence"][b+1][0]

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
        #if u not in self.memory["words"]:
        #    self.memory["words"].append(u)
        
        self.pre_heart = self.heart
        self.last_input_content = x
        if "!" not in u:
            self.last_user = u
            self.user_log.append(u)
            self.user_log.pop(0)
        
        """
        if random.randint(0,4) == 0:
            print("シャッフルしました")
            self.heart = random.randint(0, len(self.memory["sentence"]) - 1) #今の気持ち(ログの座標で表される)
        """
        
        if is_active_learning:
            self.learnSentence(x, u)
            if x == "!bad" and self.memory["sentence"][self.heart+1][0] != "!bad":
                self.memory["sentence"].insert(self.heart+1, ["!bad", "!"])
            if x == "!good" and self.memory["sentence"][self.heart+1][0] != "!good":
                self.memory["sentence"].insert(self.heart+1, ["!good", "!"])
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