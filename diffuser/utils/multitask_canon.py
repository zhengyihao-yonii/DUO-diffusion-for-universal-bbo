"""多任务实验命名：同一组任务不因命令行顺序不同而拆成多个目录。"""
from __future__ import annotations

import os


def returns_cond_path_infix(args) -> str:
    """
    同时启用 returns_condition 与 include_returns 时返回路径片段，插在 ``..._eps{eps}`` 与 ``/seed`` 之间，
    使扩散 checkpoint 与无 returns 实验分开（网络含 returns_mlp，权重不可混用）。
    ``generated_datasets`` 的轨迹 .pkl 仍与无 returns 共用；proxy / VAE 路径不变。
    默认 ``_retcond``；可用环境变量 ``GTG_RETCOND_PATH_INFIX`` 覆盖。
    """
    rc = getattr(args, "returns_condition", None)
    ir = getattr(args, "include_returns", None)
    if rc is True and ir is True:
        return os.environ.get("GTG_RETCOND_PATH_INFIX", "_retcond")
    return ""


def canonical_task_list(task_names: list[str]) -> list[str]:
    """字典序排序后的任务名列表（去重保留首次出现顺序用 sort 即可，重复任务应已在入口过滤）。"""
    return sorted(t.strip() for t in task_names if t and t.strip())


def canonical_train_tasks_csv(train_tasks: str) -> str:
    """
    将逗号分隔的训练任务字符串规范为「排序后逗号连接」。
    单任务时原样返回去首尾空白后的字符串。
    """
    parts = [t.strip() for t in train_tasks.split(",") if t.strip()]
    if len(parts) <= 1:
        return parts[0] if parts else ""
    return ",".join(sorted(parts))


def multitask_path_token(train_tasks: str) -> str:
    """用于 multi_<token>_frac... 路径段：单任务为任务名，多任务为排序后用下划线连接。"""
    csv = canonical_train_tasks_csv(train_tasks)
    if not csv:
        return ""
    if "," not in csv:
        return csv
    return "_".join(csv.split(","))
