import himawaria
import random
import re
import sys
import time 
from discord.ext import tasks
import discord
import asyncio
import datetime
dt = datetime.datetime.now()

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

intents = discord.Intents.all()
client = discord.Client(intents=intents)
mode = himawaria_instance.get_settings()["defaultMode"]
TOKEN = himawaria_instance.get_settings()["discToken"]

print("mode: {}".format(mode))
print("sentences: {}".format(len(himawaria_instance.get_memory()["sentence"])))

def setMode(x):
    global mode
    mode = x
    print("mode: {}".format(mode))

kaisu = 0
async def speak(result):
    global channel, people, mode, pin, lastMessage, messages, kaisu, dt, add, i, yukou
    try:
        print("{}: {}".format(himawaria_instance.get_settings()["myname"], result))
        pattern = re.compile(r"^!command")
        print("users: {}".format(people))
        results = result.split("\n")

        Message = ""
        for result in results:
            if bool(pattern.search(result)):
                if is_active_learning:
                    himawaria_instance.record()
                nxt = himawaria_instance.nextSpeak()
                com = result.split(" ")
                if com[1] == "discMove":
                    if not pin:
                        if client.get_channel(int(com[2])) != None:
                            channel = client.get_channel(int(com[2]))
                        else:
                            print("チャンネルが存在しません")
                            himawaria_instance.receive("チャンネルが存在しません", "!system", add=add)
                            messages.append(["チャンネルが存在しません", "!system"])
                    else:
                        print("エラー: あなたは固定されています。")
                        himawaria_instance.receive("エラー: あなたは固定されています。", "!system", add=add)
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
                elif com[1] == "saveMyData":
                    himawaria_instance.saveData()
            else:
                Message += result + "\n"
        Message = Message[:-1]
        if Message != "":
            async with channel.typing():
                yukou = True
                if len(Message) / 6 / mode >= 1:
                    if len(Message) / 6 / mode >= 5:
                        await asyncio.sleep(5)
                    else:
                        await asyncio.sleep(len(Message) / 6 / mode)
                else:
                    await asyncio.sleep(1)
                if yukou:
                    await channel.send(Message)
                    himawaria_instance.record()
                    nxt = himawaria_instance.nextSpeak()
                    if nxt:
                        print("続きを返信します")
                        await speak(nxt)
                
    except:
        himawaria_instance.receive("エラー: チャンネルがNoneか、このチャンネルに入る権限がありません", "!system", add=add)
        #messages.append(["エラー: チャンネルがNoneか、このチャンネルに入る権限がありません", "!system"])
        print("エラー: チャンネルがNoneか、このチャンネルに入る権限がありません")

# 起動時に動作する処理
@client.event
async def on_ready():
    # 起動したらターミナルにログイン通知が表示される
    global lastMessage, messages
    print('ログインしました')
    cron.start()
    himawaria_instance.receive("通知: 貴方は目を覚ましました。", "!system", add=add, reply=True)
    lastMessage = ["通知: 貴方は目を覚ましました。", "!system"]
    messages.append(["通知: 貴方は目を覚ましました。", "!system"])
    dt_now = datetime.datetime.now()
    himawaria_instance.receive(dt_now.strftime('%Y / %m / %d %H : %M : %S'), "!systemClock")
    lastMessage = [dt_now.strftime('%Y / %m / %d %H : %M : %S'), "!systemClock"]
    messages.append([dt_now.strftime('%Y / %m / %d %H : %M : %S'), "!systemClock"])

ii = 0
i = 0
add = True
# メッセージ受信時に動作する処理
@client.event
async def on_message(message):
    global pin, channel, people, lastMessage, messages, helpMessage, lastUsername, ii, mode, i, add, dt, yukou

    if bool(re.search("休んで(良い|いい)(わ|よ|わよ)|終了して|exit bot", message.content)) and "モジホコリ、" in message.content:
        exit()
        return

    if message.channel == channel or bool(re.search(himawaria_instance.himawari_instance.get_settings()["mynames"], message.content)) or isinstance(message.channel, discord.DMChannel):
        username = message.author.display_name.split("#")[0]
        if message.channel != channel:
            try:
                print("チャンネルを移動しました: {}".format(message.channel.name))
                himawaria_instance.receive("!command discMove {} | チャンネル名: {}, カテゴリー: {}, トピック: {}".format(message.channel.id, message.channel.name, message.channel.category, message.channel.topic).replace("\n", " "), username)
            except:
                print("チャンネルを移動しました: {}のDM".format(username))
                himawaria_instance.receive("!command discMove {} | 誰のDMか: {}".format(message.channel.id, username).replace("\n", " "), username)
            channel = message.channel
            people = [[himawaria_instance.himawari_instance.get_settings()["myname"], 0]]
        if message.author == client.user:
            return
        
        ff = False
        parts = message.content.split("\n")
        taisho = ""
        if bool(re.search("(.*?)--", parts[0])):
            taisho = parts[0].split("--")[0]
            ff = True
        for part in parts:
            if bool(re.search("(.*?)===(.*?)", part)):
                if taisho == "" or taisho in himawaria_instance.himawari_instance.get_settings()["mynames"]:
                    if part.split("===")[0] == "":
                        himawaria_instance.learnSentence(lastMessage[0], "!input", directLearning=True)
                        himawaria_instance.learnSentence(part.split("===")[1], "!output", directLearning=True)
                    else:
                        himawaria_instance.learnSentence(part.split("===")[0], "!input", directLearning=True)
                        himawaria_instance.learnSentence(part.split("===")[1], "!output", directLearning=True)
                ff = True
        if bool(re.search("(.*?)\n==>\n(.*?)", message.content)):
            if taisho == "" or taisho in himawaria_instance.himawari_instance.get_settings()["mynames"]:
                himawaria_instance.learnSentence(message.content.split("\n==>\n")[0], "!input", directLearning=True)
                himawaria_instance.learnSentence(message.content.split("\n==>\n")[1], "!output", directLearning=True)
            ff = True
        if ff:
            himawaria_instance.learnSentence("!good", "!system", directLearning=True)
            return
        
        ff = False
        xx = message.content.split("\n")
        for x in xx:
            if bool(re.search("(.+):- (.+)", x)):
                himawaria_instance.learnSentence(x.split(":- ")[1], x.split(":- ")[0], directLearning=True)
                ff = True
        if ff:
            himawaria_instance.learnSentence("!good", "!system", directLearning=True)
            return

        pss = []
        for ps in people:
            pss.append(ps[0])
        if username not in pss:
            people.append([username, 0])
        if message.content == "" and message.attachments == []:
            return
        if message.content == None:
            return        
        additional = ""
        for attachment in message.attachments:
            additional += "\n" + attachment.url
            print(attachment.url)
        message.content += additional

        if bool(re.search("沈黙モード|黙|だま", message.content)) and bool(re.search(himawaria_instance.himawari_instance.get_settings()["mynames"]+"|モジホコリ、", message.content)):
            himawaria_instance.receive("!command setMode 0", username)
            setMode(0)
            return
        if bool(re.search("寡黙モード|静かに|しずかに", message.content)) and bool(re.search(himawaria_instance.himawari_instance.get_settings()["mynames"]+"|モジホコリ、", message.content)):
            himawaria_instance.receive("!command setMode 1", username)
            setMode(1)
            return
        if bool(re.search("通常モード|喋って|話して|しゃべって|はなして", message.content)) and bool(re.search(himawaria_instance.himawari_instance.get_settings()["mynames"]+"|モジホコリ、", message.content)):
            himawaria_instance.receive("!command setMode 2", username)
            setMode(2)
            return
        if bool(re.search("饒舌モード", message.content)) and bool(re.search(himawaria_instance.himawari_instance.get_settings()["mynames"]+"|モジホコリ、", message.content)):
            himawaria_instance.receive("!command setMode 3", username)
            setMode(3)
            return
        if bool(re.search("ピン|じっとしてて", message.content)) and bool(re.search(himawaria_instance.himawari_instance.get_settings()["mynames"]+"|モジホコリ、", message.content)):
            pin = True
            return
        if bool(re.search("アンピン|動いていい", message.content)) and bool(re.search(himawaria_instance.himawari_instance.get_settings()["mynames"]+"|モジホコリ、", message.content)):
            pin = False
            return
        elif bool(re.search("ヘルプを表示|ヘルプ表示|show help", message.content)) and bool(re.search(himawaria_instance.himawari_instance.get_settings()["mynames"]+"|モジホコリ、", message.content)):
            await channel.send(helpMessage)
            return
        if bool(re.search("セーブして", message.content)) and bool(re.search(himawaria_instance.himawari_instance.get_settings()["mynames"]+"|モジホコリ、", message.content)):
            himawaria_instance.receive("!command saveMyData", username)
            print("セーブします")
            himawaria_instance.saveData()
            print("完了")
            return
        
        print("受信: {}, from {}".format(message.content, username))
        yukou = False
        if len(people) <= 2 or isinstance(message.channel, discord.DMChannel):
            himawaria_instance.receive(message.content, username, force=True)
        else:
            himawaria_instance.receive(message.content, username, force=True)
        lastMessage = [message.content, username]
        lastUsername = username
        i = 0
        add = True
        messages.append([message.content, username])
        dt = datetime.datetime.now()

done_zhihou = False
zhihou_span = 0
@tasks.loop(seconds=4)
async def cron():
    global people, lastMessage, messages, mode, channel, i, add, dt, done_zhihou, zhihou_span
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
        if himawaria_instance.himawari_instance.get_settings()["myname"] not in pss:
            people.append([himawaria_instance.himawari_instance.get_settings()["myname"], 0])

        if mode == 1:
            if len(messages) != 0:
                if himawaria_instance.get_current_voice() != None:
                    if bool(re.search(himawaria_instance.himawari_instance.get_settings()["mynames"], lastMessage[0])) or isinstance(channel, discord.channel.DMChannel):
                        result = himawaria_instance.speakFreely(add=add)
                        if result == None:
                            pass
                        else:
                            await speak(result)
                    messages = []
            if random.randint(0, 60*25) == 0 and himawaria_instance.get_current_voice() != None:
                result = himawaria_instance.speakFreely(add=add)
                if result == None:
                    pass
                else:
                    await speak(result)
        elif mode == 2:
            if len(messages) != 0:
                pss = []
                for ps in people:
                    pss.append(ps[0])
                aaa = ""
                for person in pss:
                    if person == himawaria_instance.himawari_instance.get_settings()["myname"]:
                        pass
                    else:
                        aaa = aaa + person + "|"
                aaa = aaa[0:-1]
                
                if len(people)-1 <= 0:
                    denominator = 0
                else:
                    denominator = len(people)-1
                if bool(re.search(himawaria_instance.himawari_instance.get_settings()["mynames"], lastMessage[0])) or isinstance(channel, discord.channel.DMChannel) or ((not bool(re.search(aaa, lastMessage[0])) or aaa == "") and random.randint(0, denominator) == 0 and himawaria_instance.get_current_voice() != None):
                    result = himawaria_instance.speakFreely(add=add)
                    if result == None:
                        pass
                    else:
                        await speak(result)
                messages = []
        elif mode == 3:
            if len(messages) != 0:
                pss = []
                for ps in people:
                    pss.append(ps[0])
                aaa = ""
                for person in pss:
                    if person == himawaria_instance.himawari_instance.get_settings()["myname"]:
                        pass
                    else:
                        aaa = aaa + person + "|"
                aaa = aaa[0:-1]
                
                if len(people)-2 <= 0:
                    denominator = 0
                else:
                    denominator = len(people)-2
                if bool(re.search(himawaria_instance.himawari_instance.get_settings()["mynames"], lastMessage[0])) or isinstance(channel, discord.channel.DMChannel) or ((not bool(re.search(aaa, lastMessage[0])) or aaa == "") and random.randint(0, denominator) == 0 and himawaria_instance.get_current_voice() != None):
                    result = himawaria_instance.speakFreely(add=add)
                    if result == None:
                        pass
                    else:
                        await speak(result)
                messages = []
        if dt_now - dt >= datetime.timedelta(seconds=20):
            if i > -2:
                i -= 1
            add = True
            if i <= -2:
                add = False

            dt = datetime.datetime.now()
            himawaria_instance.receive("!command ignore", himawaria_instance.get_last_user(), add=add)
            pattern = re.compile(r"(0|3)0 : [0-9][0-9]$")
            if bool(pattern.search(dt_now.strftime('%Y/%m/%d %H:%M:%S'))):
                himawaria_instance.receive(dt_now.strftime('%Y/%m/%d %H:%M:%S'), "!systemClock", add=add)
            print("沈黙を検知")
            if len(people)-2 <= 0:
                denominator = 0
            else:
                denominator = len(people)-2
            if mode == 2 and random.randint(0, denominator) == 0:
                result = himawaria_instance.speakFreely(add=add)
                if result == None:
                    pass
                else:
                    await speak(result)
        
    except:
        import traceback
        traceback.print_exc()


client.run(TOKEN)