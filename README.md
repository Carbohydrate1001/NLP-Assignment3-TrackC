# NLP Assignment 3 — Track C

Jiabao Gao · 122090121

本项目在 GSM8K 上评估冻结的 Qwen2.5-0.5B-Instruct，通过修改指令、选择示例和调整解码方式完成 R0–R4 迭代。最终采用 R2 提示词与固定 5 次采样，dev 准确率为 56%，test 为 43%。

## 查看结果

| 内容 | 文件 |
|---|---|
| 英文报告与源码 | `report/report.pdf`、`report/main.tex` |
| 中文要求说明 | `作业要求与完成情况.pdf` |
| 实验汇总 | `STATUS.md`、`runs/main/complete_results.csv` |
| 逐题回答 | `runs/main/R0.json`–`R4.json`、`ablation_fixed5.json`、`final_test.json` |
| 提示词搜索记录 | `runs/main/search_history.json`、`proposal_*.json` |
| 交互式查看入口 | `TrackC_Run.ipynb` |

Notebook 默认读取保存的结果。原始实验保存在已封存的 `runs/main`；复现结果写入独立目录。

## 安装与复现

运行环境为 Python 3.10+ 和 CUDA GPU。本机使用 RTX 5070 Ti、PyTorch 2.9.1+cu128。所有命令均在项目目录执行；已有相同版本的 PyTorch 时可跳过第一步。

```bash
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python reproduce_dev.py --rounds R0 R1 R2 R3 R4 ablation_fixed3 ablation_fixed5
```

`reproduce_dev.py` 从保存的配置重新评估 dev，创建带时间戳的 reproduction 目录。这一路径无需 improver API。模型和数据 revision、种子及环境版本见 `runs/main/manifest.json` 和 `environment.json`。硬件、CUDA kernel 或批次形状的差异可能影响采样输出。

## 开展新的实验

新实验统一使用 `--run-dir runs/new_experiment`。下表按执行顺序列出各阶段；计划文件和错误记录均位于这个新目录中。Colab 可将目录放在 Drive。

| 阶段 | 命令 | 准备工作 |
|---|---|---|
| R0 | `python run_trackc.py baseline --run-dir runs/new_experiment` | 自动准备 100 道 dev 和 20 道示例 |
| R1 | `python run_trackc.py human --run-dir runs/new_experiment` | 阅读 R0 错误，填写 `human_config.json` |
| R1 消融 | `python ablate_human.py --run-dir runs/new_experiment` | 完成 R0、R1 |
| R2 | `python run_trackc.py search --run-dir runs/new_experiment` | 配置 improver API |
| R3 | `python run_trackc.py retrieval --run-dir runs/new_experiment` | 分析 R2 错误，填写 `retrieval_plan.json` |
| R4 | `python run_trackc.py adaptive --run-dir runs/new_experiment` | 选择父方案，填写 `adaptive_plan.json` |
| 采样消融 | `python run_trackc.py ablation --run-dir runs/new_experiment` | 完成 R4 |
| 选定方案 | `python run_trackc.py freeze --run-dir runs/new_experiment` | 完成全部 dev 迭代与消融 |
| 最终测试 | `python run_trackc.py final --run-dir runs/new_experiment` | 完成冻结 |

R1 的计划包含 `instruction`、`demo_ids`、`failure_source`、`evidence_ids` 和 `rationale`。R3、R4 的计划记录错误来源、题目 ID 和修改理由；R4 还通过 `base_round` 指定父方案。本次 R3 从 48% 降至 36%，所以 R4 沿用 R2 的静态示例提示词，改用自适应采样。

R3 保持 R2 的示例数量，以 TF-IDF 检索替换固定选择；该方法要求 R2 至少使用一个示例。R4 先生成一个 greedy 答案和两个采样答案，没有两个有效答案相同时再追加两个。预测取出现次数最多的答案，平票取最早出现的答案。采样参数为 temperature=0.7、top-p=0.95、top-k=20，repetition_penalty=1.1。

`freeze` 在九个完整 dev 候选中依次按准确率、较少输出 tokens、列出顺序选优，并保存代码与数据哈希，锁定该目录的 dev 配置。`final` 首次执行评估选中方案，后续调用读取保存的测试结果；中断记录保留在 `test_started.json` 中。

## API 配置

本次搜索使用 [DeepSeek 官方 API](https://api-docs.deepseek.com/)，配置如下：

```text
IMPROVER_BASE_URL=https://api.deepseek.com
IMPROVER_MODEL=deepseek-flash
```

密钥通过 `IMPROVER_API_KEY` 环境变量读取。Windows 支持进程变量和同名用户变量；Colab 可从 Secrets 读取后放入 `os.environ`。Base URL 填 API 根地址，脚本会补上 `/chat/completions`。

请求使用 JSON 输出模式并关闭 thinking。每轮搜索最多调用 8 次，失败调用也计入预算。`proposal_*.json` 保存响应、模型名和 token 用量。由于 API 采样具有随机性，新一轮搜索可能产生不同提示词。

`run_local.ps1` 适用于原工作区的已配置环境，默认使用 `runs/main`。移动项目后采用上面的通用安装方法。Transformers 4.57.3 加载 tokenizer 时会查询元数据，因此本次环境保留网络连接。

## 实现与教程的关系

项目沿用教程的目标模型、数据划分、R0–R2 流程、8 次搜索预算，以及搜索阶段 200、完整评估 256 的 token 上限。在此基础上增加 R3/R4、两组消融和逐题日志。

数值提取规则为“最后一个 #### 后的数字，否则取最后一个数字”；计分采用 Decimal 精确比较，替代教程的两位小数舍入。搜索反馈随当前最佳提示词更新。最终只对选中的配置进行 test 评估。

`run_trackc.py` 管理阶段流程，`harness.py` 实现提示词、计分、检索和投票。`analyze_results.py` 汇总结果；`update_report.py` 从保存的 JSON 生成报告表格和指标宏。

## 提交

报告已署名 **Jiabao Gao（122090121）**。提交 Python 实现与报告，内容包括 R0–R4、各轮案例、RSI 曲线、最终 test 分数、dev−test 差距和消融分析。代码、写作、实验与分析各占 25 分。

[课程官网](https://nlp-course-cuhksz.github.io/#coursework-at-a-glance)要求在 [Blackboard](https://bb.cuhk.edu.cn/) 提交，并要求源码存放在 GitHub。进入课程的 Assignment 3 页面，按页面要求上传报告和代码附件，附上 GitHub 仓库链接，点击 Submit 后保存提交回执。

截止时间存在差异：附带作业 PDF 第 4 页写 **2026-12-11 23:59**，课程官网写 **2026-12-04 23:59**，均为北京时间。提交前以 Blackboard 本次作业页面及助教公告确认的日期为准。

建议文件名为 `122090121_JiabaoGao_Assignment3_Report.pdf` 和 `122090121_JiabaoGao_Assignment3_Code.zip`；作业 PDF 未指定命名格式。代码包包含报告源码与实验记录，适合解压后放入课程指定的 GitHub 仓库。模型权重及运行环境按安装步骤另行准备。

修改报告后，在 report 目录运行 `tectonic main.tex`，将生成的 `main.pdf` 更新为 `report.pdf`。
