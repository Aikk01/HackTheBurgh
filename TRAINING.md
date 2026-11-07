# RL アドバイザーの学習手順（RTX 5090 の PC 用）

6人卓ノーリミット・ホールデムの助言 AI を PPO（強化学習）で学習します。
学習が終わると `data/rl/policy.npz` ができ、Play タブの「AI アドバイス」がこのモデルの判断を表示します。

## 何を学習するか

- 6人卓の1席を担当し、プリフロップからリバーまですべての判断（フォールド / チェック・コール / 4種類のレイズ / オールイン）を学びます。
- 対戦相手は毎セッション（50〜300ハンド）ごとに入れ替わり、いろいろなスタイルの相手を引きます。
  - タイト/ルース × 強気/受け身をランダムに設定した Equity ボット
  - コーリングステーション、マニアック、ランダム、TAG、LAG
  - 過去の自分（スナップショット、相手席の約25%）
- 入力には、そのセッションで観察した相手ごとのプレイスタイル推定（参加率・レイズ率・攻撃率・ベットに降りる率・ショーダウン到達率と4タイプの確率）が入ります。相手に合わせて打ち方を変えることを学ぶための入力です。
- 報酬: 1ハンドの損益（ビッグブラインド単位）。ゲーム理論の均衡ではなく、「この相手から多く勝つ」ことを学びます。

## 1. 準備（初回だけ、10分程度）

Python 3.11 以上と git が必要です。

```bash
git clone -b poker-mirror https://github.com/Aikk01/HackTheBurgh.git
cd HackTheBurgh
python -m venv .venv
# Windows:  .venv\Scripts\activate      Linux/macOS:  source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128   # RTX 50xx は CUDA 12.8 版が必須
pip install -r requirements-train.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

最後の行で `True` と `NVIDIA GeForce RTX 5090` が出れば OK です。

## 2. 動作確認（1〜2分）

```bash
python -m pokersim rl-train --smoke
```

`EVAL upd 3 ...` と `max updates reached` が出れば成功です（smoke は本番のモデルを上書きしません）。

## 3. 本番の学習（朝まで）

終了したい時刻までの時間を `--hours` に入れます。例: 23:00 開始で 7:30 に止めたいなら 8.5。
時間が来ると自動で最終評価をして止まります（評価に数分かかるので少し余裕を持たせてください）。

```bash
python -m pokersim rl-train --hours 8.5
```

- 自分対戦（ハンドのシミュレーション）は CPU で並列に動きます。既定は「CPU コア数 − 2」個のプロセスです。
  GPU はネットワークの更新に使います。速度は主に CPU のコア数で決まります。
- 画面に1更新ごとに進捗が出ます: `upd 120 · 7,864,320 steps · ... · rollout -12.3 bb/100 · ent 1.21 ...`
  - `rollout bb/100`: 学習中の対戦成績（100ハンドあたりの獲得 BB）。上がっていけば学習が進んでいます。
  - `ent`: 行動のばらつき。少しずつ下がるのが正常です。
- 50更新ごとに評価が出ます:
  `EVAL upd 50: mean +8.1 bb/100 · tight_passive +15.2 (base +3.1, no-style +9.8) · ...`
  - 固定の5種類の卓（タイト受け身、ルース受け身、ルース強気、タイト強気、混合）で、RL モデルと既存の Equity ボットを同じ配札で比較します。
  - `no-style` は相手スタイルの入力を消した場合の成績です。`RL − no-style` が相手の読みを使えている度合いです。
  - 成績が過去最高を更新するたびに `data/rl/policy.npz` と `data/rl/policy.json` に書き出します。
- 学習の記録は `data/rl/runs/<日時>/` に残ります（`log.csv`, `eval.json`, `latest.pt`, `best.pt`）。

### 途中で止まった・止めた場合

10分ごとに `latest.pt` を保存しているので、続きから再開できます。

```bash
python -m pokersim rl-train --resume data/rl/runs/<日時>/latest.pt --hours 3
```

### よく使うオプション

| オプション | 既定 | 内容 |
|---|---|---|
| `--hours` | 9.5 | 学習時間（時間） |
| `--workers` | コア数 − 2 | 自分対戦のプロセス数 |
| `--steps-per-update` | 65536 | 1更新あたりの判断数 |
| `--eval-every` | 50 | 評価の間隔（更新回数） |
| `--eval-hands` | 3000 | 評価で各卓・各プレイヤーが打つハンド数 |
| `--p-self` | 0.25 | 相手席のうち過去の自分が座る割合 |
| `--cpu` | | GPU を使わない |

## 4. 終わったら（結果を持ち帰る）

```bash
git add data/rl/policy.npz data/rl/policy.json data/rl/runs/*/eval.json data/rl/runs/*/log.csv data/rl/runs/*/final_eval.json
git commit -m "Add trained RL advisor policy"
git push
```

`.pt`（チェックポイント）は大きいので git には入れません（.gitignore 済み）。
デモ用の PC で `git pull` して `python -m pokersim serve` を再起動すると、Play タブの助言が RL モデルになります。
デモ側の PC には PyTorch は不要です（NumPy だけで動きます）。
