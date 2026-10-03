import sys
import time 
import datetime
import random
import re
from fluxer import Bot, Message
import asyncio
import himawaria

# 1. himawari_instance の初期化処理
himawaria_instance = None
if len(sys.argv) > 1 and sys.argv[1]:
    himawaria_instance = himawaria.Himawaria(sys.argv[1])
else:
    himawaria_instance = himawaria.Himawaria("main")
time.sleep(1)

# グローバル変数の初期化
people = [[himawaria_instance.get_settings()["myname"], 0]]
channel = None
lastMessage = None
lastUsername = "誰か"
messages = []
pin = False
yukou = False
ii = 0
i = 0
is_active_learning = True
kaisu = 0
dt = datetime.datetime.now()

# 2. Botインスタンスの作成 (Client ではなく Bot を使用)
bot = Bot(command_prefix="!")

helpMessage = f"""==Himawaria ヘルプ==
このbotはユーザーのメッセージに自分の意思で返信するAIです。
話している人数に応じて返信頻度を下げます。
botの名前を呼ぶとそのチャンネルに来てくれます。
メンションでは呼べません。

=学習方法=
チャットのメッセージからも学習しますが、コマンドでの学習のほうが便利です。
ぬんへっへ！===下品だよ！

=強化学習=
「!bad」とメッセージを送ると、「このメッセージは悪い」と教えることができます。
「!good」とメッセージを送ると、「このメッセージは良い」と教えることができます。

=配慮コマンドについて=
botに「通常モード」というと「通常モード」になり、メッセージにbotの名前が含まれてなくても人数に応じて頻度を変えて返信します。また、沈黙が続いたときにメッセージを送信します。
botに「寡黙モード」というと「寡黙モード」になり、沈黙が続いたときにメッセージを送信しなくなります。
botに「沈黙モード」というと「沈黙モード」になり、呼ばれたときにしかメッセージを送信しなくなります。
botに「ピン」というと、チャンネルを動かなくなります。
botに「アンピン」というと、チャンネルを動けるようになります。
これらのコマンドのタイミングも学習します。"""

TOKEN = himawaria_instance.get_settings()["fluxerToken"]
mode = himawaria_instance.get_settings()["defaultMode"]

print("mode: {}".format(mode))
print("sentences: {}".format(len(himawaria_instance.get_memory()["sentence"])))

def setMode(x):
    global mode
    mode = x
    print("mode: {}".format(mode))

# 発言・制御関数
async def speak(result):
    global channel, people, mode, pin, lastMessage, messages, kaisu, dt, is_active_learning, i, yukou
    try:
        print("{}: {}".format(himawaria_instance.get_settings()["myname"], result))
        pattern = re.compile(r"^!command")
        print("users: {}".format(people))
        results = result.split("\n")

        Message_text = ""
        for result_line in results:
            if bool(pattern.search(result_line)):
                if is_active_learning:
                    himawaria_instance.record()
                # コマンドの解析
                com = result_line.split(" ")
                if com[1] == "fluxMove":
                    if not pin:
                        target_channel = await bot.fetch_channel(int(com[2]))
                        if target_channel != None:
                            channel = target_channel
                            try:
                                print("チャンネルを移動しました: {}".format(channel.name))
                                himawaria_instance.receive("チャンネルを移動しました: {}".format(channel.name), "!system", is_active_learning=is_active_learning)
                                people = [[himawaria_instance.get_settings()["myname"], 0]]
                                messages.append(["チャンネルを移動しました: {}".format(channel.name), "!system"])
                            except:
                                print("チャンネルを移動しました: DM")
                                himawaria_instance.receive("チャンネルを移動しました: DM", "!system", is_active_learning=is_active_learning)
                                people = [[himawaria_instance.get_settings()["myname"], 0]]
                                messages.append(["チャンネルを移動しました: DM", "!system"])
                        else:
                            print("チャンネルが存在しません")
                            himawaria_instance.receive("チャンネルが存在しません", "!system", is_active_learning=is_active_learning)
                            messages.append(["チャンネルが存在しません", "!system"])
                    else:
                        print("エラー: あなたは固定されています。")
                        himawaria_instance.receive("エラー: あなたは固定されています。", "!system", is_active_learning=is_active_learning)
                        messages.append(["エラー: あなたは固定されています。", "!system"])
                elif com[1] == "ignore":
                    pass
                elif com[1] == "setMode":
                    setMode(int(com[2]))
                elif com[1] == "saveMyData":
                    himawaria_instance.saveData()
                elif com[1] == "pin":
                    pin = True
                elif com[1] == "unpin":
                    pin = False
            else:
                Message_text += result_line + "\n"
        
        Message_text = Message_text[:-1]
        if Message_text != "":
            await channel.trigger_typing()
            yukou = True
            sleep_time = len(Message_text) / 6 / mode
            if sleep_time >= 1:
                await asyncio.sleep(min(sleep_time, 5))
            else:
                await asyncio.sleep(1)
            
            if yukou:
                await channel.send(Message_text)
                himawaria_instance.record()
                nxt = himawaria_instance.nextSpeak()
                if nxt:
                    print("続きを返信します")
                    await speak(nxt)
                
    except Exception as e:
        himawaria_instance.receive("エラー: チャンネルがNoneか、このチャンネルに入る権限がありません", "!system", is_active_learning=is_active_learning)
        import traceback
        traceback.print_exc()

# 3. 起動イベント
@bot.event
async def on_ready():
    global lastMessage, messages
    print('ログインしました')
    asyncio.create_task(cron())
        
    himawaria_instance.receive("通知: 貴方は目を覚ましました。", "!system", is_active_learning=is_active_learning, reply=True)
    lastMessage = ["通知: 貴方は目を覚ましました。", "!system"]
    messages.append(["通知: 貴方は目を覚ましました。", "!system"])
    dt_now = datetime.datetime.now()
    himawaria_instance.receive(dt_now.strftime('%Y / %m / %d %H : %M : %S'), "!systemClock")
    lastMessage = [dt_now.strftime('%Y / %m / %d %H : %M : %S'), "!systemClock"]
    messages.append([dt_now.strftime('%Y / %m / %d %H : %M : %S'), "!systemClock"])

# 4. メッセージ受信イベント
@bot.event
async def on_message(message: Message):
    global pin, channel, people, lastMessage, messages, helpMessage, lastUsername, ii, mode, i, is_active_learning, dt, yukou

    # システム終了コマンド
    if bool(re.search(r"休んで(良い|いい)(わ|よ|わよ)|終了して|exit bot", message.content)) and "モジホコリ、" in message.content:
        sys.exit()

    # DMチャンネル判定の修正 (fluxerのchannel.type等にあわせるか、簡易的に判定)
    is_dm = getattr(message.channel, 'type', None) == 'dm' or hasattr(message.channel, 'recipient')

    if message.channel == channel or bool(re.search(himawaria_instance.get_settings()["mynames"], message.content)) or is_dm:
        username = message.author.display_name.split("#")[0]
        
        if message.channel != channel:
            try:
                print("チャンネルを移動しました: {}".format(message.channel.name))
                himawaria_instance.receive("!command fluxMove {} | チャンネル名: {}, カテゴリー: {}, トピック: {}".format(message.channel.id, message.channel.name, message.channel.category, message.channel.topic).replace("\n", " "), username)
            except:
                print("チャンネルを移動しました: {}のDM".format(username))
                himawaria_instance.receive("!command fluxMove {} | 誰のDMか: {}".format(message.channel.id, username).replace("\n", " "), username)
            channel = message.channel
            people = [[himawaria_instance.get_settings()["myname"], 0]]
            
        # ボット自身のメッセージを無視
        if message.author == bot.user:
            return
        
        ff = False
        parts = message.content.split("\n")
        taisho = ""
        if bool(re.search(r"(.*?)--", parts[0])):
            taisho = parts[0].split("--")[0]
            ff = True
            
        for part in parts:
            if "===" in part:
                if taisho == "" or taisho in himawaria_instance.get_settings()["mynames"]:
                    split_part = part.split("===")
                    if split_part[0] == "" and lastMessage:
                        himawaria_instance.learnSentence(lastMessage[0], "!input", directLearning=True)
                        himawaria_instance.learnSentence(split_part[1], "!output", directLearning=True)
                    else:
                        himawaria_instance.learnSentence(split_part[0], "!input", directLearning=True)
                        himawaria_instance.learnSentence(split_part[1], "!output", directLearning=True)
                ff = True
        if bool(re.search("(.*?)\n==>\n(.*?)", message.content)):
            if taisho == "" or taisho in himawaria_instance.get_settings()["mynames"]:
                himawaria_instance.learnSentence(message.content.split("\n==>\n")[0], "!input", directLearning=True)
                himawaria_instance.learnSentence(message.content.split("\n==>\n")[1], "!output", directLearning=True)
            ff = True
        if ff:
            himawaria_instance.learnSentence("!good", "!system", directLearning=True)
            return


        if bool(re.search("沈黙モード|黙|だま", message.content)) and bool(re.search(himawaria_instance.get_settings()["mynames"]+"|モジホコリ、", message.content)):
            himawaria_instance.receive("!command setMode 0", username)
            setMode(0)
            return
        if bool(re.search("寡黙モード|静かに|しずかに", message.content)) and bool(re.search(himawaria_instance.get_settings()["mynames"]+"|モジホコリ、", message.content)):
            himawaria_instance.receive("!command setMode 1", username)
            setMode(1)
            return
        if bool(re.search("通常モード|喋って|話して|しゃべって|はなして", message.content)) and bool(re.search(himawaria_instance.get_settings()["mynames"]+"|モジホコリ、", message.content)):
            himawaria_instance.receive("!command setMode 2", username)
            setMode(2)
            return
        if bool(re.search("饒舌モード", message.content)) and bool(re.search(himawaria_instance.get_settings()["mynames"]+"|モジホコリ、", message.content)):
            himawaria_instance.receive("!command setMode 3", username)
            setMode(3)
            return
        if bool(re.search("ピン|じっとしてて", message.content)) and bool(re.search(himawaria_instance.get_settings()["mynames"]+"|モジホコリ、", message.content)):
            pin = True
            return
        if bool(re.search("アンピン|動いていい", message.content)) and bool(re.search(himawaria_instance.get_settings()["mynames"]+"|モジホコリ、", message.content)):
            pin = False
            return
        elif bool(re.search("ヘルプを表示|ヘルプ表示|show help", message.content)) and bool(re.search(himawaria_instance.get_settings()["mynames"]+"|モジホコリ、", message.content)):
            await channel.send(helpMessage)
            return
        if bool(re.search("セーブして", message.content)) and bool(re.search(himawaria_instance.get_settings()["mynames"]+"|モジホコリ、", message.content)):
            himawaria_instance.receive("!command saveMyData", username)
            print("セーブします")
            himawaria_instance.saveData()
            print("完了")
            return

        # AIの思考・返答処理をここに続行させる場合は以下を有効化
        print("受信: {}, from {}".format(message.content, username))
        yukou = False
        himawaria_instance.receive(message.content, username, force=True)
        lastMessage = [message.content, username]
        lastUsername = username
        i = 0
        is_active_learning = True
        messages.append([message.content, username])
        dt = datetime.datetime.now()


done_zhihou = False
zhihou_span = 0
async def cron():
    global people, lastMessage, messages, mode, channel, i, is_active_learning, dt, done_zhihou, zhihou_span
    while True: # 無限ループを開始
        try:
            dt_now = datetime.datetime.now()
            
            if not done_zhihou and zhihou_span <= 0:
                pattern = re.compile(r"(0|3)0 : [0-9][0-9]$")
                if bool(pattern.search(dt_now.strftime('%Y/%m/%d %H:%M:%S'))):
                    himawaria_instance.receive(dt_now.strftime('%Y/%m/%d %H:%M:%S'), "!systemClock")
                done_zhihou = True
                zhihou_span = 90
            if zhihou_span > 0:
                zhihou_span -= 1

            a = []
            for person in people:
                if person[1] < (60*10)/4:
                    a.append([person[0], person[1]+1])
            people = a
            pss = []
            for ps in people:
                pss.append(ps[0])
            if himawaria_instance.get_settings()["myname"] not in pss:
                people.append([himawaria_instance.get_settings()["myname"], 0])

            # DMチャンネル判定の修正 (fluxerの仕様に合わせる)
            is_dm = getattr(channel, 'type', None) == 'dm' or hasattr(channel, 'recipient') if channel else False

            if mode == 1:
                if len(messages) != 0:
                    if himawaria_instance.get_current_voice() != None:
                        if bool(re.search(himawaria_instance.get_settings()["mynames"], lastMessage[0])) or is_dm:
                            result = himawaria_instance.speakFreely(is_active_learning=is_active_learning)
                            if result != None:
                                await speak(result)
                        messages = []
                if random.randint(0, 60*25) == 0 and himawaria_instance.get_current_voice() != None:
                    result = himawaria_instance.speakFreely(is_active_learning=is_active_learning)
                    if result != None:
                        await speak(result)
            elif mode == 2:
                if len(messages) != 0:
                    pss = []
                    for ps in people:
                        pss.append(ps[0])
                    aaa = ""
                    for person in pss:
                        if person == himawaria_instance.get_settings()["myname"]:
                            pass
                        else:
                            aaa = aaa + person + "|"
                    aaa = aaa[0:-1]
                    
                    if len(people)-1 <= 0:
                        denominator = 0
                    else:
                        denominator = len(people)-1
                    if bool(re.search(himawaria_instance.get_settings()["mynames"], lastMessage[0])) or is_dm or ((not bool(re.search(aaa, lastMessage[0])) or aaa == "") and random.randint(0, denominator) == 0 and himawaria_instance.get_current_voice() != None):
                        result = himawaria_instance.speakFreely(is_active_learning=is_active_learning)
                        if result != None:
                            await speak(result)
                    messages = []
            elif mode == 3:
                if len(messages) != 0:
                    pss = []
                    for ps in people:
                        pss.append(ps[0])
                    aaa = ""
                    for person in pss:
                        if person == himawaria_instance.get_settings()["myname"]:
                            pass
                        else:
                            aaa = aaa + person + "|"
                    aaa = aaa[0:-1]
                    
                    if len(people)-2 <= 0:
                        denominator = 0
                    else:
                        denominator = len(people)-2
                    if bool(re.search(himawaria_instance.get_settings()["mynames"], lastMessage[0])) or is_dm or ((not bool(re.search(aaa, lastMessage[0])) or aaa == "") and random.randint(0, denominator) == 0 and himawaria_instance.get_current_voice() != None):
                        result = himawaria_instance.speakFreely(is_active_learning=is_active_learning)
                        if result != None:
                            await speak(result)
                    messages = []
            if dt_now - dt >= datetime.timedelta(seconds=20):
                if i > -2:
                    i -= 1
                is_active_learning = True
                if i <= -2:
                    is_active_learning = False

                dt = datetime.datetime.now()
                himawaria_instance.receive("!command ignore", himawaria_instance.lastUser, is_active_learning=is_active_learning)
                pattern = re.compile(r"(0|3)0 : [0-9][0-9]$")
                if bool(pattern.search(dt_now.strftime('%Y/%m/%d %H:%M:%S'))):
                    himawaria_instance.receive(dt_now.strftime('%Y/%m/%d %H:%M:%S'), "!systemClock", is_active_learning=is_active_learning)
                print("沈黙を検知")
                if len(people)-2 <= 0:
                    denominator = 0
                else:
                    denominator = len(people)-2
                if mode == 2 and random.randint(0, denominator) == 0:
                    result = himawaria_instance.speakFreely(is_active_learning=is_active_learning)
                    if result != None:
                        await speak(result)
            
        except Exception:
            import traceback
            traceback.print_exc()
            
        # 4秒待機（これが tasks.loop(seconds=4) の代わりになります）
        await asyncio.sleep(4)


# 5. ボットのメイン起動処理
async def main():
    await bot.start(TOKEN)

if __name__ == "__main__":
    asyncio.run(main())