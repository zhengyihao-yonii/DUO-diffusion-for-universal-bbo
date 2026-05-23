#!/usr/bin/env bash
# -*- coding: utf-8 -*-
# Legacy wrapper → FS step sweep + rank table (use run_quality_exp35_fs_step_sweep.sh).
exec "$(dirname "$0")/run_quality_exp35_fs_step_sweep.sh" "$@"
