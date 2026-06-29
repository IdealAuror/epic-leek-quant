# results/

回测原始输出目录。存放每个 spec 的净值 CSV、IC 表、日志等大文件。

## 命名规范

每个 spec 一个子目录：`results/<spec_id>/`，如 `results/P1-F1-EV-2026Q2-v1/`。

子目录内建议结构：
```
results/<spec_id>/
├── nav.csv              # Q1 多头组累计净值（每月末）
├── ic_monthly.csv       # 月度 Rank IC 序列
├── ic_summary.csv       # IC 描述统计
├── quantile_groups.csv  # Q1-Q5 分组收益
├── ic_by_cap.csv        # 分市值档 IC
└── ic_by_industry.csv   # 分行业 IC
```

## gitignore 策略

本目录下的 `.csv` / `.parquet` / `.log` / `.json` 大文件**不入库**
（见根目录 `.gitignore`）。仅保留 `README.md` 类小文件。

理由：回测结果可由 `joinquant/strategies/<spec_id>.py` + spec 复现，
无需在 git 中保存二进制大文件；保留代码与 spec 即可追溯。

## 复现路径

```
research/specs/<spec_id>.md   (spec，预注册通过标准)
    ↓ Flash 执行
archive/joinquant/strategies/<spec_id>.py  (执行代码，已归档)
    ↓ 聚宽运行
results/<spec_id>/            (本目录，原始输出)
    ↓ 代码审核 + 结果判定
research/reports/<spec_id>-*.md   (审核与判定报告)
```
