# Task metadata (text conditioning)

One UTF-8 file per task: `<task_key>.txt` (e.g. `ant.txt`, `gtopx2.txt`).

When `use_text_condition=True`, strings are embedded with a frozen sentence model (default MiniLM 384-d) and added to the model alongside task one-hot / CFG. English is recommended.

## File format (UniSO Table `metadata-all`, verbatim)

Each line is one field; the full file is embedded as one string (no structured parsing).

- **Task:** first column of the paper table (e.g. `Ant`, `GTOPX 2`, `TF Bind 8`)
- **Name:** second column (e.g. `Ant Morphology`, `Cassini 2`)
- **Description:** third column (LaTeX `\makecell` line breaks flattened to one line)
- **Objective:** fourth column (same)

## UniSO metadata (current)

The `.txt` files **fully match** UniSO appendix Table `tab:metadata-all` (four columns above). Filenames still use code keys (`tfbind8.txt`, `lunar_lander.txt`, etc.).

**Backup of the pre-UniSO texts:** `METADATA_BACKUP_BEFORE_UNISO.md` and `archive_pre_uniso/*.txt`.

After changing any `.txt`, **delete the embedding cache** or you will keep old vectors:

```bash
rm -rf task_metadata/.cache/
```
