import numpy as np

# --- 1. 物理音声波形の合成 (60ms) ---
fs, t_w = 1000, 0.06
t_v = np.linspace(0, t_w, int(fs * t_w), endpoint=False)
def get_w(f1, f2): return (np.sin(2*np.pi*f1*t_v) + np.sin(2*np.pi*f2*t_v))/2.0

env = {"こんにちは": get_w(200,400), "やあ": get_w(150,300),
       "こんちわ":   get_w(220,420), "砕けた表現": get_w(300,500)}
vocab, a_dim = list(env.keys()), len(t_v)

# --- 2. 皮質カラムインフラ (E/I分離・正数仕様) ---
Ne, Ni = 40, 10
np.random.seed(42)
W_in = np.random.uniform(0.1, 0.4, (Ne, a_dim))
W_ee = np.random.uniform(0.0, 0.1, (Ne, Ne))
W_ei = np.random.uniform(0.0, 0.2, (Ne, Ni))
W_ie = np.random.uniform(0.1, 0.3, (Ni, Ne))
np.fill_diagonal(W_ee, 0)

# 【弓状束可塑性配線】および【喉の筋肉マップ】
W_arc = np.random.uniform(0.01, 0.03, (Ne, Ne))
W_mus = np.random.uniform(0.01, 0.04, (a_dim, Ne))

# --- 3. 1本の連続時間ストリームによる三要素STDP学習 ---
print("--- 🧠 連続時間ストリーム学習中 (約30秒) ---")
for epoch in range(15):
    for q, a in [("こんにちは", "やあ"), ("こんちわ", "砕けた表現")]:
        stream = np.concatenate([env[q], np.zeros(10), env[a]])
        ve, vi, re = np.zeros(Ne), np.zeros(Ni), np.zeros(Ne)
        trace = np.zeros(Ne)
        
        for t in range(len(stream)):
            re[re > 0] -= 1
            sp_e = ((ve >= 1.0) & (re == 0)).astype(float)
            sp_i = (vi >= 1.0).astype(float)
            ve[sp_e == 1.0], re[sp_e == 1.0], vi[sp_i == 1.0] = 0, 3, 0
            
            # カラム内電流計算
            I_in = W_in[:, t % a_dim] * stream[t] if t < 60 else 0
            I_e = I_in + np.dot(W_ee, sp_e) - np.dot(W_ei, sp_i)
            ve = np.clip(ve * 0.85 + I_e, 0.0, 2.0)
            vi = np.clip(vi * 0.85 + np.dot(W_ie, sp_e), 0.0, 2.0)
            trace = trace * np.exp(-1.0 / 15.0) + sp_e
            
            # 後半：自発運動とドーパミン報酬
            if t >= 70:
                W_arc += 0.02 * 1.2 * np.outer(sp_e, trace)
W_arc = np.clip(W_arc, 0.0, 2.0)
print("学習完了。\n")

# --- 4. 完全自律対話ループ ---
while True:
    try:
        inp = input("ユーザー: ").strip()
        if inp.lower() in ["exit", "終了"]: break
        if inp not in env:
            print("AI言語脳: ……（その言葉の周波数を感知できません）")
            continue
            
        # フェーズ①: 聴く (60ms)
        h_wave = env[inp]
        ve, vi, re = np.zeros(Ne), np.zeros(Ni), np.zeros(Ne)
        for t in range(60):
            re[re > 0] -= 1
            sp_e = ((ve >= 1.0) & (re == 0)).astype(float)
            sp_i = (vi >= 1.0).astype(float)
            ve[sp_e == 1.0], re[sp_e == 1.0], vi[sp_i == 1.0] = 0, 3, 0
            I_e = W_in[:, t] * h_wave[t] + np.dot(W_ee, sp_e) - np.dot(W_ei, sp_i)
            ve = np.clip(ve * 0.85 + I_e, 0.0, 2.0)
            vi = np.clip(vi * 0.85 + np.dot(W_ie, sp_e), 0.0, 2.0)
            
        # フェーズ②: 外部入力0。脳の残響だけで喉を震わせる (60ms)
        out_wave = np.zeros(a_dim)
        for t in range(60):
            re[re > 0] -= 1
            sp_e = ((ve >= 1.0) & (re == 0)).astype(float)
            sp_i = (vi >= 1.0).astype(float)
            ve[sp_e == 1.0], re[sp_e == 1.0], vi[sp_i == 1.0] = 0, 3, 0
            # 弓状束(W_arc)を電気信号が走り抜ける
            I_e = np.dot(W_arc, sp_e) + np.dot(W_ee, sp_e) - np.dot(W_ei, sp_i)
            ve = np.clip(ve * 0.85 + I_e, 0.0, 2.0)
            vi = np.clip(vi * 0.85 + np.dot(W_ie, sp_e), 0.0, 2.0)
            out_wave[t] = np.dot(W_mus[t, :], sp_e)
            
        # ロバスト誤差デコーダー (最小二乗誤差)
        ans, min_err = "……（沈黙）", float('inf')
        if np.max(np.abs(out_wave)) >= 0.001:
            for name, target in env.items():
                err = np.sum((out_wave - target) ** 2)
                if err < min_err: min_err, ans = err, name
        print(f"AI言語脳: {ans}")
        
    except KeyboardInterrupt: break
