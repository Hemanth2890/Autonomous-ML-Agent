from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


class LLMHelper:

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

    def generate(self, prompt, max_new_tokens=200):

        inputs = self.tokenizer(prompt, return_tensors="pt")

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
