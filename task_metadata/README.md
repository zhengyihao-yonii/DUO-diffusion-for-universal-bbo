# Task metadata (text conditioning)

One UTF-8 file per task: `<task_key>.txt` (e.g. `ant.txt`, `gtopx2.txt`).

When `use_text_condition=True`, strings are embedded with a frozen sentence model (default MiniLM 384-d) and added to the model alongside task one-hot / CFG. English is recommended.

## File format

- **Task:** short human-readable title (name only)
- **Type:** `discrete` or `continuous` (separate line)
- **Description:** one short paragraph (domain, design space shape, what is optimized)
- **Goal:** optimization objective in one line
- **Design space:** dimensionality and structure

Do not include separate “reward” lines or generic “offline black-box benchmark” footers.
