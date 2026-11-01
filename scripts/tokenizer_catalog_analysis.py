"""Pure analysis rules shared by tokenizer plots and the foundation-model note."""
from collections import defaultdict


def vocabulary_group_key(row):
    """Collapse only an established complete ID-to-token mapping.

    A primary rank file omits companion-file and implementation-defined added
    tokens. Even equal rank-file hashes AND equal counts cannot prove equality
    of the full vocabulary. Keep those records separate until it is measured.
    """
    if row.get("shared_vocabulary_sha256"):
        return "vocab:" + row["shared_vocabulary_sha256"]
    return "model:" + row["model_id"]


def grouped_representatives(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[vocabulary_group_key(row)].append(row)
    representatives = []
    for key, members in groups.items():
        if len({row["tokenizer_vocab_size"] for row in members}) != 1:
            raise ValueError(f"Conflicting vocabulary sizes in {key}")
        representatives.append(min(members, key=lambda r: (r["timeline_date"], r["model_id"])))
    return representatives


def popularity_ranking(rows):
    """Rank exact selected repositories; missing download counts remain unranked."""
    known = [row for row in rows if row.get("downloads_30d") is not None]
    return sorted(known, key=lambda row: (-row["downloads_30d"], row["model_id"]))
