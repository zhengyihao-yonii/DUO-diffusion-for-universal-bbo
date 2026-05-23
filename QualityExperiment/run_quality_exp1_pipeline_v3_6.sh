#!/usr/bin/env bash
# -*- coding: utf-8 -*-
# English doc: Quality exp **v3.6** — FS step sweep (500..2500 eval grid), LR=1e-5 vs v3.5 5e-5.
# 中文注释: 与 v3.5 步数扫描协议一致，仅降低 FS 学习率；结果 quality_bundle_3_6/fs_step_sweep。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/run_quality_exp36_fs_step_sweep.sh" "$@"
