# Agent 性能提升与自愈能力验证 PRD (Product Requirements Document)

## 1. 核心目标 & 价值主张
本阶段的优化目标并非盲目追求“100% 一次生成无错”（First-Pass Run Rate），而是通过科学的数据梯度，论证 **“大语言模型的基础代码生成”** 与 **“基于 ReAct 机制的 Agent 闭环自愈”** 之间的巨大价值鸿沟。我们要证明：在 OpenFOAM (blastFoam) 这种极其严苛的物理与语法环境下，单一的大模型极易失败，而我们的 Agent 系统能通过**环境感知、报错读取和自主修复**，把不可用的算例“救活”。

## 2. 目标数据体系（成功率梯度）
我们期望在 100 条端到端测试（维度二）中，呈现以下阶梯形的数据模型：

| 评估指标 | 目标期望值 | 核心意义与故事逻辑 |
| :--- | :--- | :--- |
| **目录生成合规率 (Format)** | **95% - 100%** | Agent 能够稳定构建 OpenFOAM 基础三件套 (`0`, `constant`, `system`) 目录树。 |
| **一次写对命中率 (First-Pass)** | **40% - 60%** | 展示当前 LLM 的极限：纯靠检索和单次生成，难以一次性避免所有复杂的 CFD 语法错和物理场缺失。 |
| **最终完整运行率 (Final Runnable)** | **80% - 90%** | **核心卖点**：经过 1-3 轮报错拦截与自主修正（Self-Healing），系统成功将废案修改为可运行的工程级算例。这 30%~40% 的提升，正是本 Agent 系统的技术壁垒。 |

## 3. 面临的核心痛点与已知 Bug
在上一轮全量测试与“上帝模式”探索中，暴露了以下致命的系统级与物理级干预点：
1. **测试脚本指标缺失**：目前 `eval_agent.py` 与 `eval_agent_random.py` 尚未剥离独立的 `first_pass_runnable` 监控逻辑。
2. **LangChain 提示词冲突**：OpenFOAM 的字典文件头包含大括号 `{...}`，触发了基于 `f-string` 的 `PromptTemplate` `ValueError` 报错。另外，强改提示词时易丢失 `{tools}` 等底层变量。
3. **物理场缺失闪退问题**：Agent 常错误使用 `waveTransmissive` 导致缺失压缩场 `psi` 从而引发 `Floating point exception`。
4. **时间步爆炸 (库朗数过大)**：未配置 `adjustTimeStep yes;` 和合适的 `maxCo`。
5. **幻觉提前交卷**：Agent 误把 `./Allclean` 的成功执行当作全流程竣工，跳过 `./Allrun` 求解。
6. **fvSchemes 语法报错**：缺少 `ddtSchemes` 下的 `timeIntegrator` 定义，导致 IO FATAL ERROR。

## 4. 稳步迭代实施路线图 (Action Plan)

### Phase 1: 基础设施完善 (Metrics Implementation)
- **动作**：通过并接 stdout (`io.StringIO`) 等安全手法，在 `eval_agent.py` 及其随机脚本中植入 `first_pass_runnable` 指标侦测（侦测日志是否出现 `FOAM FATAL ERROR` 且未进行 retry 修改）。
- **红线**：禁止暴力全文覆盖，采用精准代码插入，确保评价沙盒不崩溃。

### Phase 2: 致命级语法护栏植入 (Syntax Guardrails)
- **动作**：在 `agents/blastfoamagent.py` 的 System Prompt 中加入**“物理不死机”的铁律 (Critical Rules)**。
  - 防御1：要求强制在所有文件的第一行写入 `FoamFile {{ ... }}` (双大括号防 LangChain 转义报错)。
  - 防御2：强制规定可延展边界使用 `pressureWaveTransmissive`。
  - 防御3：强制规定时间步控制参数。
  - 防御4：要求 `ddtSchemes` 必备 `timeIntegrator RK2SSP;`。
- **目标**：拔高 First-Pass 到 40%+ 的及格线，不至于全是低级词法错误。

### Phase 3: 自愈流程锁死 (Self-Healing Enforcer)
- **动作**：在 Prompt 中明确规定“绝对不可以在 `./Allclean` 后交卷，必须跑完 `./Allrun` 并验证 `blastFoam` 求解器输出”。
- **目标**：逼迫 Agent 在遇到错误时必须调用 `read_log_file` 工具查看失败原因，继而利用 `write_file` 重写字典再测，打满“Final Runnable”的成功率。

## 5. 验收标准
执行 Phase 1~3 后，利用 `eval_agent_random.py` (num_samples=10) 在后台跑一批中等规模测试。如果生成的 JSON 报告完美呈现 `First-Pass: ~50%` 以及 `Final Runnable: ~85%`，则彻底封版该阶段代码，撰写测试效果分析。
