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
typer src.pipeline run reproduce `
  --dataset my_dataset `
  --input D:\datasets\votes.csv `
  --user-contributions D:\datasets\user_contributions
```

## Needed resources

Flat tables may be CSV, Parquet, a JSON array, JSONL, or NDJSON. IDs must match
across all resources.

### Votes

This is the only input required by every method. Supply it with `--input`. Each
row is one vote and requires `username`, `community`, `item_id`, Unix-seconds
`timestamp`, `vote` in `{-1,+1}`, and an item-constant `label` in `{-1,+1}` for the mod. decision.

### User contributions

These are required when causal features must be built for NVSE, SEF, or Team
Formation. They are also the default source for SEF's semantic user histories:
the pipeline converts them locally to `user_documents.parquet`, without making
Arctic Shift requests. Download the JSONL archive
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

The conversion selects up to 150 documents per user, split between recent posts
and comments in the vote-table time window. Change the cap with
`--n-user-documents N` in `reproduce`, or `--n-user-docs N` below:

```powershell
typer src.pipeline run prepare user-documents `
  --input-dir D:\datasets\user_contributions `
  --users-from D:\datasets\votes.csv `
  --output data/interim/my_dataset/auxiliary/user_documents.parquet
```

## Optional resources

### User account metadata

This resource is not fetched by the Arctic Shift helper. It is optional: without
it, causal features are still generated but account tenure is unavailable.

Supply `--user-metadata PATH` with a table containing `username` and numeric
Unix-seconds `account_created_utc`. Default locations are:

```text
data/processed/user_metadata.csv             # current Reddit dataset
data/processed/<dataset>/user_metadata.csv   # another dataset
```

### Arctic Shift user-history top-up

To fill users below the configured cap from Arctic Shift, add `--download`:

```powershell
typer src.pipeline run prepare user-documents `
  --input-dir D:\datasets\user_contributions `
  --users-from D:\datasets\votes.csv `
  --output data/interim/my_dataset/auxiliary/user_documents.parquet `
  --download
```

In `reproduce`, use `--download-user-documents`. Without these flags, user
histories stay fully local.

### Moderated-item texts and scores

One command downloads both optional moderated-item tables:

```powershell
typer src.pipeline run prepare auxiliary `
  --section moderated-items `
  --input-csv D:\datasets\votes.csv `
  --votes D:\datasets\votes.csv `
  --output-dir data/interim/my_dataset/auxiliary
```

It creates:

- `post_texts.parquet`: moderated-post text for SEF and NormVio; pass an existing
  table with `--post-texts`.
- `moderated_posts_scores.parquet`: moderated-post Reddit scores for BL4/BL5;
  pass an existing table with `--external-scores`.

JSONL scores (see "User Contributions", above) describe users' historical contributions, so they cannot replace
the BL4/BL5 table. BL1-BL3 do not require external scores.

## Models

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
| `--user-contributions DIR` | Local directory of per-user JSONL files; also used to derive `user_documents.parquet`. |
| `--download-user-documents` | Top up users below the configured document cap through Arctic Shift. Default: disabled. |
| `--n-user-documents N` | Maximum documents per user for semantic embeddings. Default: 150. |
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
| `--section NAME` | Fetch `post-texts`, `user-documents`, `reddit-scores`, `moderated-items` (texts + scores), or `all`. |
| `--input-csv PATH` | Table providing `item_id`, `username`, and `timestamp`. |
| `--votes PATH` | Table providing `item_id`, `community`, and `label` for Reddit scores. |
| `--output-dir DIR` | Destination for fetched Parquet files and the acquisition manifest. |
| `--dry-run` | Print the acquisition command without executing it. |

### `prepare user-documents`

| Option | Meaning |
|---|---|
| `--input-dir DIR` | Directory containing one contribution JSONL per user. |
| `--users-from PATH` | Votes table selecting usernames and the relevant timestamp window. |
| `--output PATH` | Destination `user_documents.parquet`. |
| `--n-user-docs N` | Maximum documents per user, split between posts and comments. Default: 150. |
| `--download` | Top up users below the local limit through Arctic Shift. Default: disabled. |
| `--force` | Replace an existing locally generated Parquet. |
| `--dry-run` | Print the preparation commands without executing them. |
