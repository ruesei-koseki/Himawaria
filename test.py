import numpy as np

# 1. テキストデータと文字辞書の準備
text = "私はうんこがしたいのです。◆"
chars = sorted(list(set(text)))
vocab_size = len(chars)
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

# ハイパーパラメータの設定
N_res = 200        # リザバーのサイズ（ニューロン数）
spectral_radius = 0.95  # 記憶の寿命（残響の強さ、1未満にする）
reg = 1e-4        # 出力層を学習するときの安定化用（リッジ回帰）

# 2. ランダムなネットワーク（リザバー）の初期化（※学習中も一切変更しない固定の壁）
np.random.seed(42)
W_in = np.random.uniform(-0.5, 0.5, (N_res, vocab_size))
W_res = np.random.uniform(-0.5, 0.5, (N_res, N_res))

# 記憶が無限に増幅して暴走しないように、行列のスケール（スペクトル半径）を調整
radius = max(abs(np.linalg.eigvals(W_res)))
W_res *= (spectral_radius / radius)

# 3. 学習用データの作成（One-Hotベクトル化）
X_train = np.zeros((len(text)-1, vocab_size))
Y_train = np.zeros((len(text)-1, vocab_size))
for t in range(len(text)-1):
    X_train[t, char_to_idx[text[t]]] = 1.0     # 現在の文字
    Y_train[t, char_to_idx[text[t+1]]] = 1.0   # 次の文字（正解ラベル）

# 4. 【リザバー層の駆動】文字を順番にプールに入れて「波紋（状態）」を記録する
states = np.zeros((len(X_train), N_res))
s = np.zeros((N_res, 1))  # 初期状態はゼロ

for t in range(len(X_train)):
    u = X_train[t:t+1, :].T  # 入力文字ベクトル
    # リザバーの基本数式（過去の残響 s と新しい入力 u を混ぜ合わせて、tanhで歪ませる）
    s = np.tanh(np.dot(W_res, s) + np.dot(W_in, u))
    states[t, :] = s.ravel()

# 5. 【出力層の学習】最後の「読み取り方」を一発の最小二乗法で計算（バックプロパゲーションなし！）
# 状態の履歴（states）から正解（Y_train）を導くための行列 W_out を数学的に一発で求める
W_out = np.dot(np.dot(Y_train.T, states), np.linalg.inv(np.dot(states.T, states) + reg * np.eye(N_res)))

# 6. 【文章生成（推論）】覚えたプールを使って、新しい文字を次々に生み出す
print("--- 文章生成スタート ---")
current_char = "私"  # 最初の文字
generated_text = current_char
s = np.zeros((N_res, 1))  # プールをリセット

for _ in range(50):  # 50文字生成してみる
    # 現在の文字をベクトル化
    u = np.zeros((vocab_size, 1))
    u[char_to_idx[current_char]] = 1.0
    
    # リザバーを1ステップ動かす
    s = np.tanh(np.dot(W_res, s) + np.dot(W_in, u))
    
    # 出力層で次にくる文字の確率（らしさ）を計算
    out = np.dot(W_out, s)
    
    # 最も確率が高い文字を決定（推論）
    next_idx = np.argmax(out)
    next_char = idx_to_char[next_idx]
    
    generated_text += next_char
    current_char = next_char

    if next_char == "◆":
        break

print(generated_text)
