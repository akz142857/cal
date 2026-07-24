# Cal 实验结果格式

所有新实验摘要使用 JSON 对象，顶层至少包含：

- `result_schema_version`：当前为 `1`；
- 实验类型对应的配置、种子、指标和输出路径；
- `provenance`：源码与运行环境溯源。

`provenance` 包含：

- `source_sha256`：按文件名和内容组合得到的项目源码摘要；
- `source_files`：逐文件 SHA-256；
- `git_commit`、`git_dirty` 和 `git_status`；
- Python、PyTorch、NumPy、PyYAML 与操作系统版本；
- UTC 记录时间。

即使仓库还没有 Git 提交，源码摘要也能唯一标识实际运行内容。配置原文继续随每个
训练目录保存为 `config.yaml`，种子同时记录在配置和摘要中。

结果索引可通过以下命令重建：

```bash
uv run cal-index --results results
```

索引写入 `results/INDEX.json`，只引用机器可读摘要，不复制大型检查点。`results/`
默认不提交版本库；需要长期保留的核心数字和失败解释必须同步写入
`docs/experiments/`。
