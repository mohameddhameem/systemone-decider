"""Download and prepare the HotpotQA dataset for experiments."""

import argparse
import logging
from pathlib import Path

import pandas as pd
from datasets import load_dataset

log = logging.getLogger(__name__)

CONFIG = "distractor"


def extract_supporting_context(example: dict) -> str:
    """Return the concatenated sentences marked as supporting facts."""
    title_to_sents = dict(zip(example["context"]["title"], example["context"]["sentences"]))

    parts = []
    for title, sent_id in zip(example["supporting_facts"]["title"], example["supporting_facts"]["sent_id"]):
        sents = title_to_sents.get(title, [])
        if 0 <= sent_id < len(sents):
            parts.append(sents[sent_id].strip())

    return " ".join(parts)


def build_full_context(example: dict) -> str:
    """Concatenate all context paragraphs into a single string with titles as headers."""
    parts = []
    for title, sentences in zip(example["context"]["title"], example["context"]["sentences"]):
        para = " ".join(s.strip() for s in sentences)
        parts.append(f"[{title}] {para}")
    return "\n\n".join(parts)


def count_invalid_supporting_facts(example: dict) -> int:
    """Count supporting facts whose title or sentence index is not in the context."""
    title_to_sents = dict(zip(example["context"]["title"], example["context"]["sentences"]))
    return sum(
        not 0 <= sent_id < len(title_to_sents.get(title, []))
        for title, sent_id in zip(example["supporting_facts"]["title"], example["supporting_facts"]["sent_id"])
    )


def flatten_example(example: dict) -> dict:
    """Flatten a raw HotpotQA example into a dict suitable for a DataFrame.

    The raw paragraph/sentence structure and supporting-fact pairs are kept so that
    supporting-fact predictions can be scored with the official evaluator and paragraph
    sets can be rebuilt (e.g. with one gold paragraph removed).
    """
    sp_titles = list(example["supporting_facts"]["title"])
    return {
        "id": example["id"],
        "question": example["question"],
        "answer": example["answer"],
        "type": example["type"],
        "level": example["level"],
        "supporting_context": extract_supporting_context(example),
        "full_context": build_full_context(example),
        "num_context_paragraphs": len(example["context"]["title"]),
        "num_supporting_facts": len(sp_titles),
        "context_titles": list(example["context"]["title"]),
        "context_sentences": [list(s) for s in example["context"]["sentences"]],
        "sp_titles": sp_titles,
        "sp_sent_ids": list(example["supporting_facts"]["sent_id"]),
        "gold_titles": list(dict.fromkeys(sp_titles)),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/hotpotqa"), help="output directory")
    parser.add_argument("--sample", type=int, default=None, help="random subset size per split")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jsonl", action="store_true", help="also write JSON Lines next to Parquet")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args.out.mkdir(parents=True, exist_ok=True)

    log.info(f"Loading HotpotQA ({CONFIG}) …")
    dataset = load_dataset("hotpotqa/hotpot_qa", name=CONFIG)

    for split_name, split_data in dataset.items():
        if args.sample is not None and args.sample < len(split_data):
            split_data = split_data.shuffle(seed=args.seed).select(range(args.sample))

        records = [flatten_example(ex) for ex in split_data]
        df = pd.DataFrame(records)
        log.info(f"{split_name}: {len(df)} examples | types={df['type'].value_counts().to_dict()} | levels={df['level'].value_counts().to_dict()}")

        invalid = sum(count_invalid_supporting_facts(ex) for ex in split_data)
        if invalid:
            log.warning(f"  {invalid} supporting facts point outside the given context")

        base = args.out / f"{CONFIG}_{split_name}"
        df.to_parquet(base.with_suffix(".parquet"), index=False, engine="pyarrow")
        log.info(f"  Saved {base}.parquet")
        if args.jsonl:
            df.to_json(base.with_suffix(".jsonl"), orient="records", lines=True, force_ascii=False)
            log.info(f"  Saved {base}.jsonl")

    log.info("Done.")


if __name__ == "__main__":
    main()
