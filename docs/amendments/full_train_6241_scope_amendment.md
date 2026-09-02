# Vision-OPD 6241 全量训练范围修订

日期：2026-09-02

本修订只改变 Day 10 之后的现行执行范围，不回写 Day 1–9 的结果、哈希、失败记录或判断。旧的 E-D10-001 与 configs/vopd_1024.yaml 保留为 1024 条方案的历史准入证据，但不得启动训练。

## 现行合同

- 数据源固定为 yuanqianhao/Vision-OPD-6K revision eb5c1c2e7b9a7b6a619efe4161c7369c71bf8af4。
- 原始 train.jsonl 必须同时满足 6,241 行、4,566,587 bytes 和 SHA256 8ad2fb81da0f6fba1766545dc5f84cc2250e48704738757461b2d75aa31821df。
- 现行 split 只有 train=6,241；eval/test/retention/holdout 均为 0。
- Day 2 的 eval-128 与 retention-64 文件不改写，只在新配置中登记为 historical_only；其中样本仍属于 train-6241。
- global batch 8 使用 6,241 个真实样本加 7 个 sample_weight=0 的确定性补齐实例，共 781 步；真实样本不重复、不遗漏，补齐实例不得产生梯度或进入有效样本均值。
- 训练从冻结 Base 冷启动。Day 6 外部结果已被查看，因此原 training_design_lock 不能宣称覆盖新 6K 设计；本次变化依据数据范围、存储扩容和尾批可整除性，不依据分数调参。

## 数据恢复规则

仓库当前缺少 Day 2 所述 candidate_manifest.jsonl，因此不能假定它仍可复用。必须从同一 revision 重新取得原始 train.jsonl，通过三重身份校验后重新生成 candidate manifest，并验证历史 1,216 个 sample_id 均可在新 candidate 集合中找到。任一身份或 sample_id 对账失败都停止。

## 评测边界

新 checkpoint 不使用历史 eval-128 或 retention-64 做选优、早停或能力结论。R3 外部评测的 V* 主结果仍采用官方 191 分母；基于 train-6241 新 overlap 审计得到的分层或去重统计只能作为次级诊断，不替换 R3 主结果。
