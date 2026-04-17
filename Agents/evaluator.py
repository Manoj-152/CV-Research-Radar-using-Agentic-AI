import os
import torch
import json
import re
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

CHROMA_PATH = "chroma_db"
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")


def extract_json(raw_output):
    matches = re.findall(r"\{.*?\}", raw_output, re.DOTALL)
    if matches:
        try:
            data = json.loads(matches[-1])
            if 'score' in data:
                data['score'] = float(data['score'])
            return data
        except (json.JSONDecodeError, ValueError):
            return None
    return None


def load_prompt(filename):
    with open(os.path.join(PROMPTS_DIR, filename), 'r') as f:
        return f.read()


class EvaluatorAgent:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="BAAI/bge-large-en-v1.5",
            model_kwargs={"device": "cuda"},
            encode_kwargs={"normalize_embeddings": True}
        )

    def _get_rag_context(self, abstract):
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=self.embedding_model)
        results = db.similarity_search_with_score(abstract, k=2)

        context_str = ""
        min_distance = float('inf')
        for i, (doc, score) in enumerate(results):
            source = os.path.basename(doc.metadata.get("source", "unknown"))
            context_str += f"\n[Baseline Context {i+1} (Source: {source}, Dist: {score:.3f})]: {doc.page_content}\n"
            min_distance = min(min_distance, score)

        del db
        return context_str, round(min_distance, 3)

    def _load_llm(self):
        if self.model is None:
            # print("--> Loading LLM in 4-bit quantization.")
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                quantization_config=bnb_config,
                device_map='auto'
            )

    def _run_stage1(self, paper_title, abstract, rag_context):
        template = load_prompt("extractor_prompt.txt")
        prompt = template.format(
            rag_context=rag_context,
            paper_title=paper_title,
            abstract=abstract
        ).strip()

        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=300,
                do_sample=False
            )

        prompt_length = inputs['input_ids'].shape[-1]
        response = self.tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True)
        result = extract_json(response)

        if result is None:
            print("  Stage 1 parse failed, using fallback extraction.")
            result = {
                "contribution_type": "unknown",
                "key_claim": abstract[:300],
                "reported_gain": None,
                "compared_to": None
            }
        return result

    def _run_stage2(self, paper_title, extraction, social_score, rag_min_dist, max_retries=3):
        template = load_prompt("scorer_prompt.txt")
        prompt = template.format(
            contribution_type=extraction.get("contribution_type", "unknown"),
            key_claim=extraction.get("key_claim", ""),
            reported_gain=extraction.get("reported_gain", None),
            compared_to=extraction.get("compared_to", None),
            rag_min_dist=rag_min_dist,
            social_score=social_score,
            paper_title=paper_title
        ).strip()

        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")

        for attempt in range(max_retries):
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=150,
                    do_sample=True,
                    temperature=0.25
                )

            prompt_length = inputs['input_ids'].shape[-1]
            response = self.tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True)
            result = extract_json(response)

            if result is not None:
                return result
            print(f"  Stage 2 attempt {attempt + 1}/{max_retries}: JSON parse failed, retrying...")

        print("  Stage 2 failed all retries, returning None.")
        return None

    def evaluate(self, paper_title, abstract, social_score):
        rag_context, rag_min_dist = self._get_rag_context(abstract)

        torch.cuda.empty_cache()
        gc.collect()

        self._load_llm()

        # print("  [Stage 1] Extracting contribution...")
        extraction = self._run_stage1(paper_title, abstract, rag_context)
        # print(f"  Extraction: {extraction}")

        # print("  [Stage 2] Scoring...")
        result = self._run_stage2(paper_title, extraction, social_score, rag_min_dist)

        return result, extraction


if __name__ == '__main__':
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from test_cases.evaluator_test_papers import test_papers

    evaluator = EvaluatorAgent()

    print("\n--- Running Evaluator Test ---")
    for p in test_papers:
        res, extraction = evaluator.evaluate(p['title'], p['abs'], p['social'])
        print("#" * 100)
        print(f"\nResult for {p['title']}:")
        print(f"  Extraction : {extraction}")
        print(f"  Score      : {res}")
        print("#" * 100)
