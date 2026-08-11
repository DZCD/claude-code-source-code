# AGENTS.md

## SDK 变更必须同步文档

当 `packages/agent-sdk` 的语法、公开 API 形状、默认运行时语义、事件名称或模型侧工具契约发生变化时，必须在同一次变更中同步更新文档，不得留作后续补做：

- `packages/agent-sdk/README.md` — 包级文档与示例
- `packages/docs/src/content/docs/` — 文档站对应页面，中英文目录（含 `zh/`）都要更新
- `reference/public-api.mdx` — 公开 API 参考（新增/修改导出时更新 Functions 与 Types 表）
- 涉及行为变化的示例代码也要一并核对

文档是 SDK 接口的一部分。改完代码后运行 `bun run build`（packages/docs）确认文档站构建通过。

## 提交代码必须携带版本变更

每次提交涉及 `packages/agent-sdk` 的代码变更时，必须在同一次提交中携带版本变更，不得把版本号留到发布前再补：

- 按 semver 更新 `packages/agent-sdk/package.json` 的 `version`：新特性升 minor，修复升 patch，破坏性变更升 major。
- 同步文档站的特性可用性表：`packages/docs/src/content/docs/reference/public-api.mdx` 与 `zh/` 下对应文件，给新特性加上最低版本行。
- 在特性实际被文档化的页面（README 小节、概念页）加内联版本标记，格式为 `*Requires X.Y.Z or later.*` / `*需要 X.Y.Z 及以上版本。*`——只改 API 参考表不够，直接访问概念页的读者看不到它。
- 版本号要从 git 历史核实（特性引入的提交 + 该提交时的包版本），不要凭记忆填写。

## 处理下游反馈：先核实，再判断职责，最后选最小修法

收到 SDK 使用方的问题/需求反馈时，不照单全收，按三步走：

1. **先核实事实**。逐条对照源码和文档验证反馈的每个论断（类型形状、控制流、文档原文），反馈里常有夸大或失实——引用删节、把边缘 case 说成普遍行为、把文档已写明的内容说成缺失。回复时明确指出不实的部分，不让失实论据进入决策。
2. **再判断是不是 SDK 的职责**。只有 SDK 独占的边界才该由 SDK 修：loop 控制流出口、provider 请求/响应边界、tracing 实现、会话状态的读写原语。宿主自己一行代码能解决的、或与 SDK 已有通道功能重复的，不加——并告诉对实现有的做法。
3. **最后选最小的修法**。优先推广既有机制，不新增平行概念；能改文档解决的不改代码；一个 API 只覆盖已发生的真实需求，不为想象的场景预建通用性。结论分「认可 / 部分认可 / 不认可」三档给出理由，不为了讨好而答应。

## API 设计纪律

- **保持通道正交**：事件流服务实时 UI，`ContextTracer` 承载完整 transcript 与响应元数据，`HistoryStore` 只管会话恢复（role/content 是契约下限，元数据不进来）。新需求先问"该进哪条已有通道"，不轻易加新字段、新事件、新通道。
- **工厂风格**：公开对象一律经 `createXxx`/`defineXxx` 工厂创建，构造器不导出（`Agent` 是 type-only 导出）；端口类型只暴露方法，配置（如 `failOnError`）在工厂创建时绑定，不作为公开字段。
- **破坏性变更可以做**，但必须在特性可用性表和对应文档页标注 breaking 与迁移方式；0.x 阶段随 minor 走。



改代码前先建分支，**不在主干上直接改**。一个分支对应一件事：一个新特性，或一个修复。做完并验证通过，再合并回主干。

- 收到「加个功能」「修个 bug」类的请求，**动手写代码之前**先 `git checkout -b <分支名>`，不要先改了再想起来建分支。
- 分支名反映这次改的是什么（如 `feat/case-classification`、`fix/mask-region`）。
- **分支内每完成一个相对完整的版本就提交一次**，不要攒到最后一次性提交。一层能独立跑通、测试齐了，就是一个可提交的版本（例如「OCR 端口 + 适配器 + 缓存 + 测试」是一个版本，之后的「智能体」是下一个）。阶段性提交是默认动作，不用每次问。
- **合并回主干、push 到远端仍然要先问。**
- 提交前跑校验：imdt-claw 是 `pnpm typecheck` + `pnpm test`，涉及外部服务时加 `pnpm smoke`。GitLab CI 在 push 时跑同一组校验（`imdt-claw/.gitlab-ci.yml`，与 imdt-backend 同一套约定：zhdev 基础镜像 + TCR 镜像仓库）；main 分支自动构建两个镜像推 TCR——服务 `zhdev/imdt-claw` 和文档站 `zhdev/imdt-claw-docs`（Mintlify 站点 `docs/`：CI 的 `docs-export` job 用 `mint export` 生成静态站点走 artifact，npm cache 交 GitLab cache 只慢第一次——CI 出口逐 tarball 15s+，不能装进 docker build；`docker/Dockerfile.docs` 只是 COPY site 的运行时镜像，stack 里 `imdt-claw-docs` 服务暴露 3001，网关按独立子域名转发：测试 `imdt-claw-docs-test.zhonghuimedical.com` / 生产 `imdt-claw-docs.zhonghuimedical.com`，export 资源是绝对路径、挂子路径会 404），deploy 手动触发——ansible + docker stack 滚动更新（`ansible.yaml` + `docker/docker-compose.yml.j2`），应用环境变量用 `C_` 前缀的 CI 变量注入。
- 一次性的探针脚本（probe_*）不进仓库，验证价值该沉淀成 smoke 测试。