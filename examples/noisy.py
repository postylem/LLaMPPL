import llamppl as llp
import numpy as np
import re
from Levenshtein import distance as lev

def is_similar_levenshtein(threshold=1):
    """Whether Levenshtein distance is (at or) below threshold"""
    def f(token, s):
        if isinstance(token, llp.Token):
            token = str(token)
        if isinstance(s, llp.Token):
            s = str(s)
        # if s != '' and s[0] != token[0]:
            # return False
        if lev(token, s) <= threshold:
            return True
        else:
            return False
    return f


class NoisyTokenModel(llp.Model):
    def __init__(self, prompt, string, is_similar=is_similar_levenshtein(1)):
        super().__init__()
        self.s = prompt
        self.ctx = self.new_context(self.s)
        self.remaining_tokens = self.llama.tokenize(string)
        self.is_similar = is_similar

    def step(self):
        true_token = self.remaining_tokens.pop(0)

        token = self.sample(
            llp.Transformer(self.ctx),
            proposal=self.proposal(true_token)
        )

        self.s += token

        if len(self.remaining_tokens) == 0:
            self.observe(llp.Transformer(self.ctx), llp.EOS)
            self.finish()

    def proposal(self, true_token):
        lm_logits = self.ctx.logits()
        mask = np.array([
            0.0 if self.is_similar(true_token, v) else -np.inf
            for v in self.vocab()])
        logprobs = llp.lognormalize(lm_logits + mask)
        return llp.TokenCategorical(logprobs)

if __name__ == "__main__":
    # Create the model
    llp.LLaMAConfig.set_model_path(input("Path to GGML LLaMA model weights: "))
    model = NoisyTokenModel("", "ello, how art yo going today?")
    # Run SMC
    for i,p in enumerate(llp.smc_steer(model, 5, 15, verbose=False)):
        print(f"Particle {i}: {p} (weight {p.weight})")
else:
    llp.LLaMAConfig.set_model_path(llp.WEIGHTS_PATH)