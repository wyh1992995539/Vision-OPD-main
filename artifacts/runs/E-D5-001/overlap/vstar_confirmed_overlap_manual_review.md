# V* Bench 与 Vision-OPD 训练集：4 对确认重叠样本人工核验清单

用途：请逐对打开两张 **完整图**，核验它们是否为同一张底图。项目训练图可能额外带有局部标注框；这正是像素差异集中的区域。

审计结论：四对样本均来自项目 `train` split；两边图像尺寸相同、pHash Hamming distance 为 0。四个 V* Bench `source_id` 为 `24`、`59`、`131`、`161`。

## 1. V* source_id 24 ↔ 项目 train row 5308

- V* 完整图：[打开](file:///root/autodl-tmp/benchmark_data/converted/vstar/images/24_4d134bc072.png)
- 项目训练完整图：[打开](file:///root/autodl-tmp/data/vision_opd_1024/images/49bc01a8c66ea01b4ecf3cf7cea7775d.png)
- 项目训练裁剪图（标注目标）：[打开](file:///root/autodl-tmp/data/vision_opd_1024/teacher_images/002493_49bc01a8c66ea01b4ecf3cf7cea7775d.png)
- V* 题目：`What is the color of the car?`；标准答案：`B`（black）。
- 项目训练题目：`How many orange lifebuoys are visible along the railing of the top deck?`；标注答案：`D`（5）。
- 审计证据：两图均为 2248×1500，pHash distance 0；平均 RGB 差异 0.4376，局限在 `(886,652)–(1716,866)`。

## 2. V* source_id 59 ↔ 项目 train row 5315

- V* 完整图：[打开](file:///root/autodl-tmp/benchmark_data/converted/vstar/images/59_5a5b0f9b7d.png)
- 项目训练完整图：[打开](file:///root/autodl-tmp/data/vision_opd_1024/images/df688e359c2cd50b8e16652417df3562.png)
- 项目训练裁剪图（标注目标）：[打开](file:///root/autodl-tmp/data/vision_opd_1024/teacher_images/002500_df688e359c2cd50b8e16652417df3562.png)
- V* 题目：`What is the color of the cyclist's bag?`；标准答案：`A`（orange and black）。
- 项目训练题目：`What country code is visible on the blue strip of the license plate?`；标注答案：`A`（D）。
- 审计证据：两图均为 2249×1500，pHash distance 0；平均 RGB 差异 0.0407，局限在 `(768,1333)–(849,1359)`。

## 3. V* source_id 131 ↔ 项目 train row 3560

- V* 完整图：[打开](file:///root/autodl-tmp/benchmark_data/converted/vstar/images/131_e794a80eb1.png)
- 项目训练完整图：[打开](file:///root/autodl-tmp/data/vision_opd_1024/images/f703343baf1b0a0fa6e4639e439b3a88.png)
- 项目训练裁剪图（标注目标）：[打开](file:///root/autodl-tmp/data/vision_opd_1024/teacher_images/000490_f703343baf1b0a0fa6e4639e439b3a88.png)
- V* 题目：`Is the dog on the left or right side of the train?`；标准答案：`B`（right）。
- 项目训练题目：`What kind of animal is shown in the image?`；标注答案：`D`（dog）。
- 审计证据：两图均为 2250×1500，pHash distance 0；平均 RGB 差异 0.0171，局限在 `(1862,798)–(1885,828)`。

## 4. V* source_id 161 ↔ 项目 train row 3285

- V* 完整图：[打开](file:///root/autodl-tmp/benchmark_data/converted/vstar/images/161_0159a99ed2.png)
- 项目训练完整图：[打开](file:///root/autodl-tmp/data/vision_opd_1024/images/9fe981bdc6cccd78338ddc90f2dad144.png)
- 项目训练裁剪图（标注目标）：[打开](file:///root/autodl-tmp/data/vision_opd_1024/teacher_images/000178_9fe981bdc6cccd78338ddc90f2dad144.png)
- V* 题目：`Is the small red car on the left or right side of the baby carriage?`；标准答案：`A`（left）。
- 项目训练题目：`What is the black bicycle locked to?`；标注答案：`A`（trash can）。
- 审计证据：两图均为 2250×1500，pHash distance 0；平均 RGB 差异 0.1185，局限在 `(1436,1006)–(1657,1126)`。

## 来源证据

- 机器候选与路径、哈希、训练样本 ID：[overlap_candidates.jsonl](overlap_candidates.jsonl)
- 人工复核决定：[manual_review_decisions.json](manual_review_decisions.json)
- 汇总政策：[overlap_report.json](overlap_report.json)
