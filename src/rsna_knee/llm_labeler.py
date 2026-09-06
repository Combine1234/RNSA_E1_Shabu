from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from .constants import TARGETS


DEFINITIONS = (
    "ACL injury; MCL injury; medial meniscus tear; lateral meniscus tear; "
    "medial tibiofemoral osteoarthritis; lateral tibiofemoral osteoarthritis; "
    "patellofemoral osteoarthritis; joint effusion; synovitis; Baker's cyst; "
    "bone contusion; acute fracture"
)


def parse_codes(text: str) -> list[str] | None:
    normalized = str(text).upper().strip()
    match = re.search(r"(?:[YNU]\s*,\s*){11}[YNU]", normalized)
    if match:
        return re.findall(r"[YNU]", match.group(0))
    compact = re.search(r"[YNU]{12}", normalized)
    if compact:
        return list(compact.group(0))
    tokens = re.findall(r"\b[YNU]\b", normalized)
    return tokens if len(tokens) == len(TARGETS) else None


def prompt_for_report(report: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You label multilingual knee MRI radiology reports. For each finding, answer Y only when "
                "affirmed, N only when explicitly absent, and U when not addressed or uncertain. Do not infer "
                "image truth beyond the report. Output exactly one 12-character code using only Y, N, or U, "
                "with no spaces, separators, or explanation. The code must stop after character 12."
            ),
        },
        {
            "role": "user",
            "content": f"Order: {DEFINITIONS}\nReport:\n{report}\n12-character code:",
        },
    ]


def run_worker(
    input_path: str | Path,
    output_path: str | Path,
    model_path: str | Path,
    batch_size: int,
    max_input_tokens: int,
    max_new_tokens: int,
) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    ).to("cuda")
    model.eval()
    rows = [json.loads(line) for line in Path(input_path).read_text(encoding="utf-8").splitlines() if line]
    with Path(output_path).open("w", encoding="utf-8") as output:
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            prompts = [
                tokenizer.apply_chat_template(prompt_for_report(row["report"]), tokenize=False, add_generation_prompt=True)
                for row in batch
            ]
            encoded = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_input_tokens,
            ).to("cuda")
            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.eos_token_id,
                )
            response_tokens = generated[:, encoded["input_ids"].shape[1] :]
            responses = tokenizer.batch_decode(response_tokens, skip_special_tokens=True)
            for row, response in zip(batch, responses):
                codes = parse_codes(response)
                output.write(json.dumps({
                    "index": row["index"],
                    "id": row["id"],
                    "codes": codes or ["U"] * len(TARGETS),
                    "parse_ok": codes is not None,
                    "raw_response": response,
                }) + "\n")
            output.flush()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-input-tokens", type=int, default=1536)
    parser.add_argument("--max-new-tokens", type=int, default=40)
    args = parser.parse_args()
    run_worker(
        args.input, args.output, args.model, args.batch_size, args.max_input_tokens, args.max_new_tokens
    )


if __name__ == "__main__":
    main()
