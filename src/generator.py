import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

LLM_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

RAG_SYSTEM_PROMPT = (
    "You are a helpful teaching assistant for a university AI module. "
    "Answer the student's question using ONLY the information in the provided lecture excerpts. "
    "If the excerpts do not contain enough information to answer, say clearly that the lecture "
    "did not cover this. Be concise (2-4 sentences)."
)

BASELINE_SYSTEM_PROMPT = (
    "You are a helpful teaching assistant for a university AI module. "
    "Answer the student's question concisely (2-4 sentences) based on your own knowledge."
)


class Generator:
    def __init__(self, model_name: str = LLM_NAME):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        self.model.eval()

    def _generate(self, system_prompt: str, user_prompt: str, max_new_tokens: int = 200) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def answer_with_context(self, question: str, context_chunks: list[str]) -> str:
        context = "\n\n".join(f"Excerpt {i+1}: {c}" for i, c in enumerate(context_chunks))
        user_prompt = f"Lecture excerpts:\n{context}\n\nQuestion: {question}"
        return self._generate(RAG_SYSTEM_PROMPT, user_prompt)

    def answer_without_context(self, question: str) -> str:
        return self._generate(BASELINE_SYSTEM_PROMPT, question)


if __name__ == "__main__":
    gen = Generator()
    q = "What is dropout and why is it used?"
    context = [
        "One popular method is dropout, where during training we randomly switch off a fraction of "
        "neurons in each forward pass, which prevents the network from relying too heavily on any "
        "single neuron and encourages more robust, distributed representations."
    ]
    print("RAG answer:\n", gen.answer_with_context(q, context))
    print("\nBaseline answer:\n", gen.answer_without_context(q))
