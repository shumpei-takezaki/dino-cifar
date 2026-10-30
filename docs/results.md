# 実験結果: CIFAR-10 DINO vs DINOv2 (ViT-Tiny/4)

最終更新: 2026-10-03

## 1. 実験条件

- データ: CIFAR-10 train 50k(SSL学習ではラベル不使用)、評価は test 10k
- Backbone: ViT-Tiny/4(embed 192, depth 12, heads 3, 32×32 → 8×8=64 patch tokens)
- 3構成とも `configs/base.yaml` を共有(augmentation `mrnghia90`、global bs 256、AdamW lr 5e-4、1000 epochs、warmup 10、
  wd 0.04→0.4、EMA 0.996→1.0、fp32)。違いは損失の重みとregister数のみ。

| 構成 | config | 損失 | registers |
|---|---|---|---:|
| DINO | `configs/dino.yaml` | DINO | 0 |
| DINOv2 | `configs/dinov2.yaml` | DINO + iBOT(1.0) + KoLeo(0.1) | 0 |
| DINOv2-Reg4 | `configs/dinov2_reg4.yaml` | 同上 | 4 |

- seed 0 のみ(1本)。コード: commit `96a2cc0`。A100-40GB 1枚で各 約16.5時間(中央値 56〜59 s/epoch)。
- 評価はすべて EMA teacher backbone。kNN: k=20, T=0.07, L2正規化。線形分類: 標準化した凍結特徴に SGD 100 epoch(lr 0.1, cosine)、
  ハイパラは固定(未調整、§5参照)。

## 2. 最終結果(1000 epoch, CIFAR-10 test top-1 %)

| Method | Registers | kNN CLS | kNN PatchMean | Linear CLS | Linear PatchMean |
|---|---:|---:|---:|---:|---:|
| DINO | 0 | 82.04 | 78.27 | 79.86 | 77.74 |
| DINOv2 | 0 | **85.49** | **86.05** | **81.25** | **86.59** |
| DINOv2-Reg4 | 4 | 84.59 | 84.04 | 79.84 | 85.00 |

参考の下限(同じ kNN 設定で測定):

| 特徴 | kNN |
|---|---:|
| ランダム初期化 ViT-Tiny/4(CLS / PatchMean) | 30.96 / 32.18 |
| 生ピクセル(平均を引いたもの) | 45.93 |

外部の参考値として、solo-learn の CIFAR-10 DINO(ResNet18, 1000 ep)は linear 89.52%。backbone も評価方法も違うので直接は比較できない。

## 3. kNN の推移(CLS / PatchMean)

| epoch | 100 | 300 | 500 | 700 | 900 | 1000 |
|---|---|---|---|---|---|---|
| DINO | 76.7 / 74.4 | 79.9 / 77.6 | 81.6 / 77.7 | 81.7 / 77.5 | 82.0 / 77.8 | 82.0 / 78.3 |
| DINOv2 | 81.5 / 78.1 | 83.5 / 81.6 | 84.7 / 83.3 | 85.1 / 85.1 | 85.4 / 85.8 | 85.5 / 86.0 |
| DINOv2-Reg4 | 81.6 / 77.8 | 83.4 / 81.6 | 84.4 / 82.7 | 84.7 / 83.1 | 84.7 / 84.0 | 84.6 / 84.0 |

50 epoch ごとの値はすべて `outputs/<構成>/seed_0/train.jsonl`(`knn_*` キー)にある。

## 4. 学習の健全性

| 構成 | ep | dino_loss | ibot_loss | koleo_loss | grad_norm(クリップ前) | teacher CLS std |
|---|---:|---:|---:|---:|---:|---:|
| DINO | 1 / 100 / 500 / 1000 | 7.74 / 1.86 / 0.74 / 0.27 | – | – | 1.7 / 1.6 / 4.3 / 10.0 | 0.80 / 0.95 / 2.05 / 2.91 |
| DINOv2 | 1 / 100 / 500 / 1000 | 7.78 / 1.51 / 0.62 / 0.25 | 3.94 / 1.43 / 1.30 / 1.12 | 0.63 / −0.10 / −0.14 / −0.18 | 2.8 / 2.0 / 4.7 / 13.3 | 0.81 / 1.13 / 2.03 / 2.91 |
| DINOv2-Reg4 | 1 / 100 / 500 / 1000 | 7.72 / 1.45 / 0.56 / 0.23 | 3.92 / 1.36 / 1.27 / 1.15 | 0.60 / −0.10 / −0.15 / −0.18 | 5.2 / 2.0 / 4.5 / 12.0 | 0.78 / 1.14 / 2.02 / 2.88 |

- 全構成で NaN / Inf はゼロ。10 epoch 目以降、epoch 平均の total loss が前の epoch より上がった幅は最大 0.014 で、ほぼ単調に下がった。
- 崩壊の兆候なし。teacher CLS の std は単調に増え、DINO center のノルムも約15で安定している。
- KoLeo が負なのは −log(最近傍距離) の定義どおりで、距離が1を超えると負になる。
- クリップ前の grad norm は後半に増え、400 epoch あたりからクリップ上限 3.0 を常に超えている(3構成とも同じ傾向)。

## 5. 所見

1. **DINOv2 は全指標で DINO を上回った。** 差が最も大きいのはパッチ平均で、kNN +7.8pt、linear +8.9pt。
   iBOT(マスクしたパッチを予測する損失)がパッチトークンの表現を改善する、という仮説と合っている。
2. **伸び方が違う。** DINO は500 epoch あたりで頭打ちになったが、DINOv2 のパッチ平均 kNN は1000 epoch まで伸び続けた
   (500→1000 で +2.7pt)。DINOv2 は2000 epoch に延長する余地がある。
3. **DINOv2 では PatchMean ≧ CLS になった**(kNN 86.05 vs 85.49)。DINO では逆で、CLS のほうが3.8pt 高い。
4. **Reg4 は DINOv2(R=0)よりわずかに低かった**(PatchMean kNN −2.0pt)。seed が1本なので、ばらつきの範囲かどうかは判断できない。
5. **CLS では linear が kNN より低い**(3構成とも。DINOv2: 81.25 vs 85.49)。線形分類のハイパラが固定・未調整なのが原因と考えている。
   CLS の linear 値は暫定扱いにする。

## 6. 成果物

| 構成 | 出力 |
|---|---|
| DINO | `outputs/dino/seed_0/` |
| DINOv2 | `outputs/dinov2/seed_0/` |
| DINOv2-Reg4 | `outputs/dinov2_reg4/seed_0/` |

各ディレクトリの中身:
- `teacher_backbone.pt`: 下流タスク用
- `checkpoints/checkpoint_epoch_{0100..1000}.pt`: 全状態の checkpoint
- `train.jsonl`: 正規のログ
- `metrics.json`
- `config.yaml`
- `env.json`

`outputs/` は git の管理外(各ディレクトリ約1.1GB)。生の指標は [results_seed0.json](results_seed0.json) にある。

## 7. 未実施(後回し)

- 線形分類の評価方法の見直し: train から検証用データを分け、lr / wd を選び直す。test では選ばない。
- seed 1, 2 の追加と平均±標準偏差の報告。特に Reg4 との差を確かめるのに必要。
- DINOv2 の2000 epoch への延長(パッチ平均がまだ伸びていたため)。

## 8. 定性解析(seed 0、1000 epoch)

`python -m cifar_dino.analyze`(README 参照)で作成した。図は `outputs/analysis/seed_0/` にある(git 管理外)。

- **t-SNE(`tsne.png`、test 5000枚):** DINO のパッチ平均はクラスが混ざっており、まとまっているのは乗り物系だけ。
  DINOv2 と Reg4 のパッチ平均は、クラスごとのクラスタに分かれている。kNN の差(78.3 → 86.0)と合っている。
  CLS では3構成とも乗り物がよく分かれ、動物同士(cat/dog/deer)は重なっている。
- **パッチの PCA(`patch_pca.png`):** そのまま PCA にかけると、パッチ分散の12〜14%が位置だけで説明され、
  PC1 が画像の上と下を分ける成分になった(前景が取れない)。位置ごとの平均を引いてから PCA にかけると、
  DINOv2 と Reg4 では PC1 > 0 の領域が物体の形(frog、horse、cat など)に沿う。DINO は前景と背景がまだらで、物体の形にならない。
  ただし、空や水面が前景側に入る画像もある(ship、deer)。8×8 の解像度なので、目安と考えるのがよい。

## 9. パッチトークンの潜在空間と register の要否(seed 0、1000 epoch)

RAE で latent として使う前提で、test 5000枚 × 64パッチの teacher パッチトークンを調べた(2026-10-04)。
計算は使い捨てのスクリプトで行い、コードには残していない。

| | DINO | DINOv2 | DINOv2-Reg4 |
|---|---:|---:|---:|
| final LN 後のノルム | 41.2 ± 0.2 | 40.9 ± 0.2 | 40.4 ± 0.2 |
| final LN 後のチャネルごとの std | 2.1〜6.6 | 1.9〜6.4 | 1.9〜4.3 |
| 第1主成分の分散 / 分散の90%に必要な次元数 / 実効ランク | 4.6% / 129 / 126 | 4.6% / 116 / 118 | 4.3% / 111 / 115 |
| 位置だけで説明できる分散 | 13.6% | 12.1% | 11.7% |
| cos:隣のパッチ / 同じ画像の中 / 別の画像の同じ位置 | 0.61 / 0.06 / 0.15 | 0.74 / 0.15 / 0.16 | 0.74 / 0.14 / 0.17 |
| final LN 前のパッチのノルム:中央値 / 最大 | 1.3 / 2.4 | 1.9 / 3.4 | 1.9 / 3.2 |

- 192次元のほぼ全体に分散が広がっていて、低次元にまとまってはいない。
- DINO では、同じ画像の中のパッチより、別の画像の同じ位置のパッチのほうが似ている。位置の情報が強く出ている。
  DINOv2 は空間的に滑らかで、同じ画像の中のまとまりも強い。§8 のパッチ PCA の結果と合っている。
- **register は不要と判断した。** R=0 でも、final LN 前のノルムの最大は中央値の1.8倍までで、ノルムが極端に大きい
  アーティファクトトークンは無い。Reg4 の register のノルム(約1.3)は CLS と同程度。
  そこで、RAE には DINOv2(R=0)を使うことにした(実装は [rae-cifar](https://github.com/shumpei-takezaki/rae-cifar))。

## 10. エンコーダの大きさ:ViT-Tiny/4 と ViT-S/4(DINOv2、R=0、seed 0、1000 epoch)

rae-cifar でエンコーダの容量を比べるため(rae-cifar の docs/experiments/06)、DINOv2 を ViT-S/4 でも学習した
(`configs/dinov2_vits.yaml`:幅 384、ヘッド 6。ほかは `dinov2.yaml` と同じ)。出力は `outputs/dinov2_vits/seed_0/`。

| エンコーダ | 幅 | kNN CLS | kNN PatchMean | Linear CLS | Linear PatchMean |
|---|---:|---:|---:|---:|---:|
| ViT-Tiny/4 | 192 | 85.49 | 86.05 | **81.25** | 86.59 |
| ViT-S/4 | 384 | **86.96** | **87.04** | 76.59 | **89.54** |

kNN の推移(CLS / PatchMean):

| | 100 | 300 | 500 | 700 | 1000 |
|---|---|---|---|---|---|
| ViT-Tiny/4 | 81.5 / 78.1 | 83.5 / 81.6 | 84.7 / 83.3 | 85.1 / 85.1 | 85.5 / 86.1 |
| ViT-S/4 | 82.3 / 80.7 | 84.5 / 84.1 | 85.2 / 84.5 | 86.2 / 86.2 | 87.0 / 87.0 |

- **S は Tiny より良い**(kNN +1.0〜1.5pt、PatchMean の linear +3.0pt)。どの時点でも S が上で、1000 epoch でもまだ伸びている。
- **CLS の linear だけは S の方が低い**(76.59)。§5 のとおり、CLS の linear は線形分類のハイパラが固定・未調整なので暫定扱い。
- **計算精度が違う:** Tiny は厳密な fp32、S は TF32(fp32 では S は約 2.4 倍遅い)。学習時間は S が約 18 時間(A100 1 枚)。
