# Reproducibility
<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Run from the repository root with Python:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

Run the current dataset with:

```powershell
typer src.pipeline run reproduce --dataset reddit --input data/processed/final_intersection_dataset.csv
```

K-fold evaluation is not enabled by default. To run it as well:

```powershell
typer src.pipeline run reproduce --dataset reddit --input data/processed/final_intersection_dataset.csv --k-fold
```

For another dataset, choose a unique name and pass its CSV from any location:

```powershell
typer src.pipeline run reproduce --dataset my_dataset --input D:\datasets\votes.csv
```

## Needed resources

Flat tables may be CSV, Parquet, a JSON array, JSONL, or NDJSON. IDs must match
across all resources.

### Votes

This is the only input required by every method. Supply it with `--input`. Each
row is one vote and requires `username`, `community`, `item_id`, Unix-seconds
`timestamp`, `vote` in `{-1,+1}`, and an item-constant `label` in `{-1,+1}` for the mod. decision.

### User contributions

These are required only when causal features must be built for NVSE, SEF, or
Team Formation. Download the JSONL archive
[here](https://drive.google.com/file/d/1RNWxXLgP8cpWInrWxeoUFdHQ0D3GKgzR/view?usp=drive_link)
and extract it to:

```text
data/raw/user_contributions/user_contributions/   # current Reddit dataset
data/raw/<dataset>/user_contributions/            # another dataset
```

Alternatively, leave it elsewhere and pass `--user-contributions DIR`. The
directory must contain one `<username>.jsonl` per user. Each line is a JSON
object with `subreddit`, numeric `created_utc`, and `id` or canonical `name`. A
post has `title`; a comment has `body` and canonical `parent_id`. Numeric `score`
is optional:

```json
{"id":"abc123","subreddit":"example","created_utc":1700000000,"title":"Post title","score":4}
{"name":"t1_def456","subreddit":"example","created_utc":1700000100,"body":"Reply","parent_id":"t1_xyz789","score":2}
```

### User account metadata

This resource is not fetched by the Arctic Shift helper. It is optional: without
it, causal features are still generated but account tenure is unavailable.

Supply `--user-metadata PATH` with a table containing `username` and numeric
Unix-seconds `account_created_utc`. Default locations are:

```text
data/processed/user_metadata.csv             # current Reddit dataset
data/processed/<dataset>/user_metadata.csv   # another dataset
```

### Arctic Shift tables

Fetch post texts, historical user documents, and Reddit item scores once:

```powershell
typer src.pipeline run prepare auxiliary `
  --section all `
  --input-csv D:\datasets\votes.csv `
  --votes D:\datasets\votes.csv `
  --output-dir data/interim/my_dataset/auxiliary
```

These tables serve a different purpose from the contribution JSONLs. Arctic
Shift supplies the text consumed by SEF and NormVio, plus post-level Reddit
scores for BL4/BL5.

The output directory contains:

- `post_texts.parquet`: `item_id`, `text`, and also `title`, `selftext` when
  generating NVSE scores. Pass an existing file with `--post-texts`.
- `user_documents.parquet`: `username`, Unix-seconds `created_utc`, `text`.
  Pass an existing file with `--user-documents`.
- `moderated_posts_scores.parquet`: `item_id`, numeric `score`, used by
  baselines BL4/BL5. Pass an existing file with `--external-scores`.

### Models

To generate NVSE scores, download the nine NormVio checkpoints
[here](https://drive.google.com/file/d/1khNw_M4HcJOQe2gftfXMybraYChVt3T7/view?ts=6a05d126)
and place them at:

```text
external/normvio/normvio_redditmodels/<category>/finetuned_model.pt
```

The categories are `spam`, `meta-rules`, `content`, `harassment`, `hatespeech`,
`format`, `off-topic`, `trolling`, and `incivility`.

Hugging Face automatically downloads NormVio's
`DeepPavlov/bert-base-cased-conversational` and SEF's
`sentence-transformers/all-MiniLM-L6-v2` on first use. 

## CLI options

### `reproduce`

| Option | Meaning |
|---|---|
| `--input PATH` | Required votes table; mutually exclusive with `--config`. |
| `--dataset NAME` | Namespace separating splits, caches, results, and reports. Default: `reddit`. |
| `--method NAME`, `-m NAME` | Select a method; repeat for several. Omit to run all. Choices: `baselines`, `community-notes`, `var`, `nvse`, `sef`, `ma-qsmf`, `team-formation`. |
| `--post-texts PATH` | Local post-text table. |
| `--user-documents PATH` | Local historical user-document table. |
| `--user-contributions DIR` | Local directory of per-user JSONL files. |
| `--user-metadata PATH` | Local account-metadata table. |
| `--external-scores PATH` | Local per-item score table. |
| `--k-fold` | Also run five-fold evaluation and summaries. |
| `--force` | Rebuild splits, causal features, and NVSE scores. |
| `--graphs` / `--no-graphs` | Enable or skip final figures. Default: enabled. |
| `--device cpu|cuda|cuda:N` | Execution device. Default: `cpu`. |
| `--dry-run` | Validate and print the plan without executing it. |
| `--config PATH`, `-c PATH` | Optional advanced JSON configuration for non-standard model/cache paths. |

### `prepare auxiliary`

| Option | Meaning |
|---|---|
| `--section NAME` | Fetch `post-texts`, `user-documents`, `reddit-scores`, or `all`. |
| `--input-csv PATH` | Table providing `item_id`, `username`, and `timestamp`. |
| `--votes PATH` | Table providing `item_id`, `community`, and `label` for Reddit scores. |
| `--output-dir DIR` | Destination for fetched Parquet files and the acquisition manifest. |
| `--dry-run` | Print the acquisition command without executing it. |
