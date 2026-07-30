# Dataset Schema Summary

## filtered_votes.parquet

- Rows: 59098
- Columns: 21

| Column | Type | Description |
|--------|------|------------|
| username | object | Reddit username |
| community | object | Subreddit name |
| item_id | object | Identifier |
| timestamp | datetime64[ns, UTC] | Creation timestamp |
| vote | int8 | Vote (-1, +1) |
| label | int8 | Post label |
| karma | int64 | User karma |
| post_karma | float64 | User karma |
| comment_karma | int64 | User karma |
| num_posts | float64 | — |
| num_comments | int64 | — |
| tenure_days | float64 | Account age in days |
| earliest_post_at | float64 | — |
| earliest_comment_at | int64 | — |
| last_post_at | float64 | — |
| last_comment_at | int64 | — |
| active_communities | float64 | — |
| subreddit_entropy | float64 | Subreddit name |
| n_interaction_partners | float64 | — |
| total_interactions | float64 | — |
| temporal_entropy | float64 | Diversity metric (higher = more spread activity) |


## users.parquet

- Rows: 1967
- Columns: 16

| Column | Type | Description |
|--------|------|------------|
| username | object | Reddit username |
| karma | int64 | User karma |
| post_karma | float64 | User karma |
| comment_karma | int64 | User karma |
| num_posts | float64 | — |
| num_comments | int64 | — |
| tenure_days | float64 | Account age in days |
| earliest_post_at | float64 | — |
| earliest_comment_at | int64 | — |
| last_post_at | float64 | — |
| last_comment_at | int64 | — |
| active_communities | float64 | — |
| subreddit_entropy | float64 | Subreddit name |
| n_interaction_partners | float64 | — |
| total_interactions | float64 | — |
| temporal_entropy | float64 | Diversity metric (higher = more spread activity) |


## history/user_documents.parquet

- Rows: 309851
- Columns: 12

| Column | Type | Description |
|--------|------|------------|
| username | object | Reddit username |
| source | object | — |
| thing_id | object | Identifier |
| subreddit | object | Subreddit name |
| created_utc | int64 | Creation timestamp |
| title | object | Post title |
| text | object | Text content (post/comment) |
| score | int64 | Reddit score |
| num_comments | float64 | — |
| link_id | object | Identifier |
| parent_id | object | Identifier |
| url | object | — |


## history/user_flairs.parquet

- Rows: 48898
- Columns: 4

| Column | Type | Description |
|--------|------|------------|
| username | object | Reddit username |
| subreddit | object | Subreddit name |
| author_flair_text | object | Text content (post/comment) |
| count | int64 | Interaction/activity count |


## history/user_subreddit_activity.parquet

- Rows: 181106
- Columns: 6

| Column | Type | Description |
|--------|------|------------|
| username | object | Reddit username |
| subreddit | object | Subreddit name |
| count | int64 | Interaction/activity count |
| weighted_count | float64 | Interaction/activity count |
| after | object | — |
| before | object | — |


## history/user_interactions.parquet

- Rows: 193913
- Columns: 5

| Column | Type | Description |
|--------|------|------------|
| username | object | Reddit username |
| other_user | object | — |
| count | int64 | Interaction/activity count |
| after | object | — |
| before | object | — |


## history/user_history_summary.parquet

- Rows: 1967
- Columns: 9

| Column | Type | Description |
|--------|------|------------|
| username | object | Reddit username |
| n_documents | int64 | — |
| n_posts | int64 | — |
| n_comments | int64 | — |
| n_subreddit_rows | int64 | Subreddit name |
| n_interaction_rows | int64 | — |
| n_flair_rows | int64 | User flair |
| posts_truncated | bool | — |
| comments_truncated | bool | — |


## splits/test_votes.parquet

- Rows: 8876
- Columns: 21

| Column | Type | Description |
|--------|------|------------|
| username | object | Reddit username |
| community | object | Subreddit name |
| item_id | object | Identifier |
| timestamp | datetime64[ns, UTC] | Creation timestamp |
| vote | int8 | Vote (-1, +1) |
| label | int8 | Post label |
| karma | int64 | User karma |
| post_karma | float64 | User karma |
| comment_karma | int64 | User karma |
| num_posts | float64 | — |
| num_comments | int64 | — |
| tenure_days | float64 | Account age in days |
| earliest_post_at | float64 | — |
| earliest_comment_at | int64 | — |
| last_post_at | float64 | — |
| last_comment_at | int64 | — |
| active_communities | float64 | — |
| subreddit_entropy | float64 | Subreddit name |
| n_interaction_partners | float64 | — |
| total_interactions | float64 | — |
| temporal_entropy | float64 | Diversity metric (higher = more spread activity) |


## splits/train_votes.parquet

- Rows: 40900
- Columns: 21

| Column | Type | Description |
|--------|------|------------|
| username | object | Reddit username |
| community | object | Subreddit name |
| item_id | object | Identifier |
| timestamp | datetime64[ns, UTC] | Creation timestamp |
| vote | int8 | Vote (-1, +1) |
| label | int8 | Post label |
| karma | int64 | User karma |
| post_karma | float64 | User karma |
| comment_karma | int64 | User karma |
| num_posts | float64 | — |
| num_comments | int64 | — |
| tenure_days | float64 | Account age in days |
| earliest_post_at | float64 | — |
| earliest_comment_at | int64 | — |
| last_post_at | float64 | — |
| last_comment_at | int64 | — |
| active_communities | float64 | — |
| subreddit_entropy | float64 | Subreddit name |
| n_interaction_partners | float64 | — |
| total_interactions | float64 | — |
| temporal_entropy | float64 | Diversity metric (higher = more spread activity) |


## splits/val_votes.parquet

- Rows: 9322
- Columns: 21

| Column | Type | Description |
|--------|------|------------|
| username | object | Reddit username |
| community | object | Subreddit name |
| item_id | object | Identifier |
| timestamp | datetime64[ns, UTC] | Creation timestamp |
| vote | int8 | Vote (-1, +1) |
| label | int8 | Post label |
| karma | int64 | User karma |
| post_karma | float64 | User karma |
| comment_karma | int64 | User karma |
| num_posts | float64 | — |
| num_comments | int64 | — |
| tenure_days | float64 | Account age in days |
| earliest_post_at | float64 | — |
| earliest_comment_at | int64 | — |
| last_post_at | float64 | — |
| last_comment_at | int64 | — |
| active_communities | float64 | — |
| subreddit_entropy | float64 | Subreddit name |
| n_interaction_partners | float64 | — |
| total_interactions | float64 | — |
| temporal_entropy | float64 | Diversity metric (higher = more spread activity) |

