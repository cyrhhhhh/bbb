# DMRL: 反事实稀疏极性证据传输的多模态情感分析

本项目实现论文中的 **DMRL** 模型（Counterfactual Sparse Polar Evidence Transport），
用于 CMU-MOSI / CMU-MOSEI / CH-SIMS 三个多模态情感分析基准的训练与评测。

## 目录结构

```
.
├── dmrl.py          # 模型定义（编码器、稀疏证据抽取、Sinkhorn 传输、反事实路由、损失）
├── config.py        # 各数据集与模型/损失的超参数配置
├── data_loader.py   # MMSA 风格 .pkl 数据加载
├── metrics.py       # MAE / Corr / Acc-2 / Acc-5 / Acc-7 / F1 评测指标
├── train.py         # 训练入口（早停 + 测试评估）
└── requirements.txt # 依赖
```

## 环境安装

```powershell
pip install -r requirements.txt
```

如需从国内镜像下载预训练 BERT，已在 `dmrl.py` 中默认设置 `HF_ENDPOINT=https://hf-mirror.com`。

## 数据准备

使用标准的 MMSA 处理后数据（单个 `.pkl` 文件，内部包含 `train` / `valid` / `test` 三个划分）。
每个划分为一个字典，至少包含以下字段：

| 字段 | 形状 | 说明 |
| --- | --- | --- |
| `text_bert` | `[N, 3, L]` | BERT 输入：input_ids / attention_mask / token_type_ids |
| `audio` | `[N, T_a, D_a]` | 声学特征序列 |
| `vision` | `[N, T_v, D_v]` | 视觉特征序列 |
| `regression_labels` | `[N]` | 连续情感标签（MOSI/MOSEI 取值 [-3,3]，SIMS 取值 [-1,1]）|

各数据集默认特征维度（`config.py` 中的 `feature_dims = (text, audio, video)`）：

- MOSI: `(768, 5, 20)`
- MOSEI: `(768, 74, 35)`
- SIMS: `(768, 33, 709)`

> 这些公开数据集需自行从官方/MMSA 渠道获取，本仓库不包含数据。

## 训练

```powershell
python train.py --dataset mosi --data_path data\mosi\aligned_50.pkl --seed 1111
```

常用参数：

| 参数 | 说明 |
| --- | --- |
| `--dataset` | `mosi` / `mosei` / `sims` |
| `--data_path` | 处理后的 `.pkl` 文件路径 |
| `--seed` | 随机种子（论文报告多种子均值）|
| `--device` | `cuda` 或 `cpu`（默认自动检测）|
| `--batch_size` / `--learning_rate` / `--bert_lr` | 覆盖默认超参 |
| `--num_epochs` / `--early_stop` | 训练轮数与早停耐心值 |
| `--no_finetune` | 冻结 BERT 编码器 |

训练过程中以 **验证集 MAE** 做早停，并保存最优权重至 `checkpoints/`，
最后输出最优验证点对应的测试集指标。

## 评测指标

- **MAE**：平均绝对误差（越低越好）
- **Corr**：皮尔逊相关系数
- **Acc-7 / Acc-5**：多分类准确率
- **Acc-2 / F1**：二分类准确率与 F1（MOSI/MOSEI 同时给出“负/非负”与“负/正”两种约定）

## 多种子复现

```powershell
foreach ($s in 1111,1112,1113,1114,1115) {
    python train.py --dataset mosi --data_path data\mosi\aligned_50.pkl --seed $s
}
```

对多次运行的测试指标取平均即为论文报告数值。


TEST-(DLF) >>  acc_7: 0.5581  acc_5: 0.5769  acc_2: 0.8740  F1_score: 0.8742  Corr: 0.7995  MAE: 0.5036  Loss: 0.5033

TEST-(DLF) >>  acc_7: 0.4825  acc_5: 0.5583  acc_2: 0.8735  F1_score: 0.8736  Corr: 0.8390  MAE: 0.6478  Loss: 0.6478
TEST-(DLF) >>  acc_7: 0.4781  acc_5: 0.5612  acc_2: 0.8750  F1_score: 0.8751  Corr: 0.8404  MAE: 0.6498  Loss: 0.6498
