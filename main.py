import os
import json
import torch
import gc
from datetime import datetime

from Agents.fetch_arxiv import fetch_recent_papers
from Agents.evaluator import EvaluatorAgent
from Agents.synthesizer import SynthesizerAgent
from utils import get_hf_metadata

SCORE_THRESHOLD = 3.5


def get_log_dir():
    date_str = datetime.now().strftime("%Y-%m-%d")
    log_dir = os.path.join("logs", date_str)
    os.makedirs(log_dir, exist_ok=True)
    return log_dir, date_str


def load_processed_ids(log_dir):
    path = os.path.join(log_dir, "processed_ids.json")
    if os.path.exists(path):
        with open(path, 'r') as f:
            return set(json.load(f))
    return set()


def save_processed_ids(log_dir, processed_ids):
    path = os.path.join(log_dir, "processed_ids.json")
    with open(path, 'w') as f:
        json.dump(list(processed_ids), f)


def append_evaluation(log_dir, record):
    path = os.path.join(log_dir, "evaluations.jsonl")
    with open(path, 'a') as f:
        f.write(json.dumps(record) + "\n")


def load_evaluations(log_dir):
    path = os.path.join(log_dir, "evaluations.jsonl")
    if not os.path.exists(path):
        return []
    results = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def save_report(log_dir, date_str, results, briefing, total_fetched):
    report_path = os.path.join(log_dir, f"radar_{date_str}.txt")

    with open(report_path, 'w') as f:
        f.write(f"CV Research Radar — Daily Briefing\n")
        f.write(f"Date      : {date_str}\n")
        f.write(f"Fetched   : {total_fetched} papers\n")
        f.write(f"Above {SCORE_THRESHOLD} : {len(results)} papers\n")
        f.write("=" * 70 + "\n\n")

        f.write("STATE OF THE ART BRIEFING\n")
        f.write("-" * 70 + "\n")
        f.write(briefing + "\n\n")

        f.write("=" * 70 + "\n")
        f.write("FULL RESULTS (sorted by score)\n")
        f.write("-" * 70 + "\n\n")

        for i, r in enumerate(results, 1):
            f.write(f"[{i}] {r['title']}\n")
            f.write(f"    Score     : {r['score']} ({r['novelty'].upper()})\n")
            f.write(f"    Published : {r['published']}\n")
            f.write(f"    HF Upvotes: {r['social_score']}\n")
            f.write(f"    Key Claim : {r['key_claim']}\n")
            f.write(f"    Reason    : {r['reason']}\n")
            f.write(f"    PDF       : {r['pdf_url']}\n\n")

    print(f"\nReport saved to: {report_path}")
    return report_path


def run():
    print("\n" + "=" * 70)
    print("CV Research Radar — Starting Pipeline")
    print("=" * 70)

    log_dir, date_str = get_log_dir()
    print(f"Log directory: {log_dir}")

    # --- Phase 1: Fetch ---
    papers = fetch_recent_papers(days_back=2, total_papers=4)
    total_fetched = len(papers)

    processed_ids = load_processed_ids(log_dir)
    new_papers = [p for p in papers if p['id'] not in processed_ids]
    print(f"\n{len(new_papers)} new papers to evaluate ({total_fetched - len(new_papers)} already processed today).")

    if not new_papers:
        print("Nothing new to evaluate.")
        return

    # --- Phase 2: Evaluate ---
    evaluator = EvaluatorAgent()

    for i, paper in enumerate(new_papers, 1):
        print(f"\n[{i}/{len(new_papers)}] {paper['title']}")

        hf_meta = get_hf_metadata(paper['id'])
        social_score = hf_meta['upvotes'] if hf_meta else 0

        result, extraction = evaluator.evaluate(paper['title'], paper['abstract'], social_score)

        # Mark as processed immediately regardless of outcome
        processed_ids.add(paper['id'])
        save_processed_ids(log_dir, processed_ids)

        if result is None:
            print("  Evaluation failed, skipping.")
            continue

        print(f"  Score: {result['score']} | {result['novelty'].upper()}")

        # Write to disk immediately — crash-safe
        if result['score'] >= SCORE_THRESHOLD:
            record = {
                'title':        paper['title'],
                'arxiv_id':     paper['id'],
                'published':    paper['published'],
                'pdf_url':      paper['pdf_url'],
                'score':        result['score'],
                'novelty':      result['novelty'],
                'reason':       result['reason'],
                'key_claim':    extraction.get('key_claim', ''),
                'social_score': social_score
            }
            append_evaluation(log_dir, record)

        torch.cuda.empty_cache()
        gc.collect()

    # --- Phase 3: Synthesize ---
    results = load_evaluations(log_dir)

    if not results:
        print("\nNo papers passed the score threshold.")
        return

    results.sort(key=lambda x: x['score'], reverse=True)
    print(f"\n{len(results)} papers above threshold. Top: {results[0]['title']} ({results[0]['score']})")

    print("\nRunning synthesizer on top 3...")
    synthesizer = SynthesizerAgent(evaluator.tokenizer, evaluator.model)
    briefing = synthesizer.synthesize(results[:3])

    # --- Phase 4: Save Report ---
    save_report(log_dir, date_str, results, briefing, total_fetched)


if __name__ == '__main__':
    run()
