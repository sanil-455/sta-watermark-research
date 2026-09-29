import json
import re
from pathlib import Path

import spacy


BASELINE_PATH = Path("results/raw/baseline_safe.json")

SAMPLE_TO_PROMPT_ID = {
    1: 0,
    2: 1,
    3: 2,
    4: 3,
    5: 4,
}

TARGET_SAMPLES = {1, 2, 3}

OUTPUT_DIR = Path(
    "results/raw/attack2_candidates_by_sample"
)

INSERTION_WORDS = [
    "also",
    "still",
    "just",
    "quite",
    "rather",
    "generally",
    "often",
    "usually",
    "even",
    "perhaps",
    "simply",
    "mainly",
]

FORBIDDEN_DELETIONS = {
    "not",
    "no",
    "never",
    "neither",
    "nor",
    "only",
    "without",
    "hardly",
    "barely",
    "cannot",
    "can't",
    "won't",
    "isn't",
    "aren't",
    "wasn't",
    "weren't",
    "don't",
    "doesn't",
    "didn't",
}

WORD_PATTERN = re.compile(
    r"^[A-Za-z]+(?:[-'][A-Za-z]+)*$"
)


def load_baselines():
    with BASELINE_PATH.open() as f:
        return json.load(f)


def find_baseline(sample_number, baselines):
    expected_id = SAMPLE_TO_PROMPT_ID[
        sample_number
    ]

    matches = [
        x
        for x in baselines
        if int(x["prompt_id"]) == expected_id
    ]

    if len(matches) != 1:
        raise RuntimeError(
            f"Research Sample {sample_number} "
            f"must map to exactly one "
            f"prompt_id={expected_id}; "
            f"found {len(matches)}."
        )

    sample = matches[0]

    if int(sample["prompt_id"]) != expected_id:
        raise RuntimeError(
            f"ID mismatch for Research Sample "
            f"{sample_number}."
        )

    return sample


def is_word(token):
    return bool(
        WORD_PATTERN.fullmatch(token.text)
    )


def deletion_allowed(token):
    word = token.text.lower()

    if not is_word(token):
        return False

    if word in FORBIDDEN_DELETIONS:
        return False

    if token.ent_type_:
        return False

    if token.like_num:
        return False

    if token.like_url or token.like_email:
        return False

    if len(word) <= 2:
        return False

    return True


def insertion_allowed(doc, position, word):
    if position <= 0 or position >= len(doc):
        return False

    left = doc[position - 1]
    right = doc[position]

    # We only insert at an actual whitespace boundary.
    if right.idx <= 0:
        return False

    if not doc.text[right.idx - 1].isspace():
        return False

    if left.is_punct or right.is_punct:
        return False

    if left.ent_type_ or right.ent_type_:
        return False

    if right.text.startswith("'"):
        return False

    if word in {
        "also",
        "still",
        "just",
        "even",
        "perhaps",
        "simply",
    }:
        return right.pos_ in {
            "VERB",
            "AUX",
            "ADJ",
            "ADV",
        }

    if word in {"quite", "rather"}:
        return right.pos_ in {
            "ADJ",
            "ADV",
        }

    if word in {
        "generally",
        "often",
        "usually",
        "mainly",
    }:
        return right.pos_ in {
            "VERB",
            "AUX",
            "ADJ",
            "ADV",
        }

    return False


def make_deletion(text, token):
    start = token.idx
    end = token.idx + len(token.text)

    # If the word is followed by punctuation,
    # remove the punctuation too. This prevents
    # malformed forms such as "loves,, and".
    if end < len(text):
        if text[end] in ",;:":
            end += 1
            if end < len(text) and text[end].isspace():
                end += 1
            return text[:start] + text[end:]

    # Otherwise remove one adjacent whitespace.
    if end < len(text) and text[end].isspace():
        end += 1

    return text[:start] + text[end:]


def make_insertion(text, token, word):
    position = token.idx

    # Candidate is created only at a real whitespace
    # boundary, so preserve the existing space.
    return (
        text[:position]
        + word
        + " "
        + text[position:]
    )


def generate_for_sample(
    sample,
    sample_number,
    nlp,
):
    text = sample["watermarked_text"]
    doc = nlp(text)

    candidates = []

    for token in doc:
        if deletion_allowed(token):
            attacked = make_deletion(
                text,
                token,
            )

            candidates.append(
                {
                    "sample_number": sample_number,
                    "prompt_id": int(
                        sample["prompt_id"]
                    ),
                    "attack_type": "deletion",
                    "original_word": token.text,
                    "position": token.i,
                    "replacement": None,
                    "attacked_text": attacked,
                }
            )

    for position, token in enumerate(doc):
        for word in INSERTION_WORDS:
            if insertion_allowed(
                doc,
                position,
                word,
            ):
                attacked = make_insertion(
                    text,
                    token,
                    word,
                )

                candidates.append(
                    {
                        "sample_number": sample_number,
                        "prompt_id": int(
                            sample["prompt_id"]
                        ),
                        "attack_type": "insertion",
                        "original_word": None,
                        "position": position,
                        "replacement": word,
                        "attacked_text": attacked,
                    }
                )

    return candidates


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading spaCy model...")
    nlp = spacy.load("en_core_web_sm")

    baselines = load_baselines()

    for sample_number in sorted(
        TARGET_SAMPLES
    ):
        sample = find_baseline(
            sample_number,
            baselines,
        )

        prompt_id = int(
            sample["prompt_id"]
        )

        baseline_z = float(
            sample["watermarked_z"]
        )

        if baseline_z <= 2.0:
            raise RuntimeError(
                f"Research Sample {sample_number} "
                f"(prompt_id={prompt_id}) is "
                f"already undetected: "
                f"z={baseline_z}"
            )

        candidates = generate_for_sample(
            sample,
            sample_number,
            nlp,
        )

        output_path = (
            OUTPUT_DIR
            / f"sample_{sample_number}.json"
        )

        result = {
            "sample_number": sample_number,
            "prompt_id": prompt_id,
            "baseline_z": baseline_z,
            "baseline_detected": True,
            "candidate_count": len(candidates),
            "candidates": candidates,
        }

        with output_path.open("w") as f:
            json.dump(
                result,
                f,
                indent=2,
            )

        print(
            f"[✓ SAVED] Research Sample "
            f"{sample_number} "
            f"(prompt_id={prompt_id}): "
            f"{len(candidates)} candidates -> "
            f"{output_path}"
        )


if __name__ == "__main__":
    main()   
