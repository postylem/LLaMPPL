import llamppl as llp
import numpy as np
from Levenshtein import distance as lev

def is_similar_levenshtein(token, s, threshold=1):
    """Whether Levenshtein distance is (at or) below threshold"""
    if isinstance(token, llp.Token):
        token = str(token)
    if isinstance(s, llp.Token):
        s = str(s)
    if s != '' and s[0] != token[0]:
        return False
    if lev(token, s) <= threshold:
        return True
    else:
        return False

class NoisyModel(llp.Model):
    def __init__(self, string):
        super().__init__()

        self.ctx = self.new_context(self.s)
        self.remaining_tokens = self.llama.tokenize(string)
        self.is_similar = is_similar_levenshtein

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

    def proposal(self, current_token):
        lm_logits = self.ctx.logits()
        lev_similarity_mask = np.array([
            0.0 if self.is_similar(current_token, v) else -np.inf
            for v in self.vocab()])
        reweighted_logits = sum(
            [lm_logits, lev_similarity_mask]
        )
        q_logprobs = llp.lognormalize(reweighted_logits)
        return llp.TokenCategorical(q_logprobs)

if __name__ == "__main__":
    # Create the model
    llp.LLaMAConfig.set_model_path(input("Path to GGML LLaMA model weights: "))
    model = NoisyModel("Hello, how art you going today?")
    # Run SMC
    for i,p in enumerate(llp.smc_steer(model, 5, 15), verbose=False):
        print(f"Particle {i}: {p} (weight {p.weight})")
else:
    llp.LLaMAConfig.set_model_path(llp.WEIGHTS_PATH)