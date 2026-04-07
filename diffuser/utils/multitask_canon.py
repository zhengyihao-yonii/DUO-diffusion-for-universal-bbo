"""多任务实验命名：同一组任务不因命令行顺序不同而拆成多个目录。"""
from __future__ import annotations


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
