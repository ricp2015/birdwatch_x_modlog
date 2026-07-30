"""
Inspection script per T1 (caratteristiche utente).

Cosa fa:
- Trova tutte le cartelle *_by_split (uno o piu' metodi)
- Per ogni split dentro, elenca i file presenti
- Per ogni file .parquet trovato, stampa: shape, colonne+dtype, 3 righe di esempio
- Controlla la copertura di users.parquet rispetto agli utenti visti nei voti/item_scores

Uso:
    python inspect_outputs.py --base . --users users.parquet --votes votes.parquet

Se non sai i path esatti, lancia prima senza argomenti da dentro la cartella
del progetto: prova a indovinare automaticamente cercando pattern *_by_split.

L'output e' pensato per essere copiato/incollato in chat: e' compatto,
niente dump di interi dataframe.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

SEP = "=" * 70


def describe_parquet(path: Path, n_sample: int = 3) -> None:
    print(f"\n--- {path} ---")
    try:
        df = pd.read_parquet(path)
    except Exception as e:
        print(f"  [ERRORE lettura] {e}")
        return

    print(f"  shape: {df.shape}")
    print("  colonne (nome: dtype):")
    for col, dtype in df.dtypes.items():
        print(f"    - {col}: {dtype}")

    if len(df) > 0:
        print(f"  esempio ({min(n_sample, len(df))} righe):")
        with pd.option_context("display.max_columns", None, "display.width", 160):
            print(df.head(n_sample).to_string(index=False))

    # se c'e' una colonna che sembra un identificatore utente, conta gli unici
    user_like_cols = [c for c in df.columns if c.lower() in
                       ("username", "user_id", "rater_id", "voter", "voter_id", "author")]
    for c in user_like_cols:
        print(f"  n_unique[{c}]: {df[c].nunique()}")

    item_like_cols = [c for c in df.columns if c.lower() in
                       ("item_id", "post_id", "note_id")]
    for c in item_like_cols:
        print(f"  n_unique[{c}]: {df[c].nunique()}")

    sub_like_cols = [c for c in df.columns if "subreddit" in c.lower() or "community" in c.lower()]
    if sub_like_cols:
        print(f"  colonne subreddit/community trovate: {sub_like_cols}")
    else:
        print("  [NESSUNA colonna subreddit/community trovata in questo file]")


def scan_by_split_dirs(base: Path) -> list[Path]:
    return sorted(base.glob("**/*_by_split"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=str, default=".",
                     help="Cartella radice da cui cercare le cartelle *_by_split")
    ap.add_argument("--users", type=str, default=None,
                     help="Path a users.parquet (metadata Arctic Shift), opzionale")
    ap.add_argument("--votes", type=str, default=None,
                     help="Path a votes.parquet (voti grezzi), opzionale, per confronto copertura")
    ap.add_argument("--max-files-per-split", type=int, default=10,
                     help="Limite file .parquet ispezionati per ogni cartella split (evita output enorme)")
    args = ap.parse_args()

    base = Path(args.base).resolve()
    print(f"Cerco cartelle *_by_split sotto: {base}")
    by_split_dirs = scan_by_split_dirs(base)

    if not by_split_dirs:
        print("Nessuna cartella *_by_split trovata. Passa --base con il path giusto.")
    else:
        print(f"Trovate {len(by_split_dirs)} cartelle *_by_split:")
        for d in by_split_dirs:
            print(f"  - {d}")

    for method_dir in by_split_dirs:
        print(f"\n{SEP}\nMETODO: {method_dir}\n{SEP}")
        # split possono essere sottocartelle dirette, o (per i baseline) un livello in piu'
        split_dirs = [p for p in method_dir.iterdir() if p.is_dir()]
        if not split_dirs:
            print("  [vuoto]")
            continue

        for split_dir in split_dirs:
            print(f"\n  ### split: {split_dir.name}")
            all_files = sorted(split_dir.rglob("*"))
            files_only = [f for f in all_files if f.is_file()]
            print(f"  File presenti ({len(files_only)}):")
            for f in files_only:
                print(f"    {f.relative_to(split_dir)}")

            parquet_files = [f for f in files_only if f.suffix == ".parquet"]
            for f in parquet_files[: args.max_files_per_split]:
                describe_parquet(f)

            if len(parquet_files) > args.max_files_per_split:
                skipped = len(parquet_files) - args.max_files_per_split
                print(f"  [...{skipped} altri file .parquet non mostrati, alza --max-files-per-split]")

            # una sola volta per split, prova a controllare
            break  # rimuovi questo break se vuoi ispezionare TUTTI gli split, non solo il primo

    # --- controllo copertura users.parquet ---
    if args.users:
        users_path = Path(args.users)
        print(f"\n{SEP}\nCONTROLLO COPERTURA: {users_path}\n{SEP}")
        try:
            users_df = pd.read_parquet(users_path)
            print(f"users.parquet shape: {users_df.shape}")
            print(f"colonne: {list(users_df.columns)}")
            id_col = next((c for c in users_df.columns
                           if c.lower() in ("username", "user_id")), None)
            if id_col is None:
                print("  [ATTENZIONE] nessuna colonna username/user_id riconosciuta automaticamente")
            else:
                n_users_meta = users_df[id_col].nunique()
                print(f"  utenti unici in users.parquet: {n_users_meta}")

                if args.votes:
                    votes_df = pd.read_parquet(args.votes)
                    vote_id_col = next((c for c in votes_df.columns
                                         if c.lower() in ("username", "user_id")), None)
                    if vote_id_col:
                        all_voters = set(votes_df[vote_id_col].unique())
                        covered = all_voters & set(users_df[id_col].unique())
                        missing = all_voters - set(users_df[id_col].unique())
                        print(f"  utenti totali nei voti: {len(all_voters)}")
                        print(f"  coperti da metadata: {len(covered)} "
                              f"({100 * len(covered) / len(all_voters):.1f}%)")
                        print(f"  NON coperti: {len(missing)} "
                              f"({100 * len(missing) / len(all_voters):.1f}%)")
                    else:
                        print("  [ATTENZIONE] nessuna colonna username/user_id trovata in votes.parquet")
        except Exception as e:
            print(f"  [ERRORE lettura users.parquet] {e}")
    else:
        print("\n(--users non passato: salto il controllo di copertura metadata)")

    print(f"\n{SEP}\nFINE. Copia/incolla questo output in chat.\n{SEP}")


if __name__ == "__main__":
    main()