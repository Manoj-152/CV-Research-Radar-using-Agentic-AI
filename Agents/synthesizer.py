import os
import torch

PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")


def load_prompt(filename):
    with open(os.path.join(PROMPTS_DIR, filename), 'r') as f:
        return f.read()


class SynthesizerAgent:
    def __init__(self, tokenizer, model):
        # Reuses the already-loaded model from EvaluatorAgent to avoid double loading
        self.tokenizer = tokenizer
        self.model = model

    def synthesize(self, top_papers):
        """
        top_papers: list of dicts with keys — title, score, novelty, reason, key_claim
        Returns the synthesized briefing as a string.
        """
        papers_block = ""
        for i, p in enumerate(top_papers, 1):
            papers_block += (
                f"[{i}] {p['title']}\n"
                f"    Score     : {p['score']} ({p['novelty'].upper()})\n"
                f"    Key Claim : {p['key_claim']}\n"
                f"    Evaluation: {p['reason']}\n\n"
            )

        template = load_prompt("synthesizer_prompt.txt")
        prompt = template.format(papers_block=papers_block.strip()).strip()

        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=400,
                do_sample=True,
                temperature=0.4
            )

        prompt_length = inputs['input_ids'].shape[-1]
        response = self.tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True)
        return response.strip()
