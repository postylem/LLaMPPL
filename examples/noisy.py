import llamppl as llp
import numpy as np
import re
from Levenshtein import distance as lev


def is_similar_levenshtein(threshold=1):
    """Make predicate for whether Levenshtein distance is (at or) below threshold"""

    def f(s, token):
        if isinstance(token, llp.Token):
            token = str(token)
        if isinstance(s, llp.Token):
            s = str(s)
        if isinstance(s, list):
            s = "".join(str(x) for x in s)
        # if s != '' and s[0] != token[0]:
        # return False
        if lev(token, s) <= threshold:
            return True
        else:
            return False

    return f


def split_before_whitespace(string):
    """
    Simple whitespace-preserving word-tokenizer: splits string every location
    that follows a non-whitespace char and precedes a whitespace char.
    E.g.: 'Here, an example.' -> ['Here,', ' an', ' example.']
    """
    return re.split(r"(?<=\S)(?=\s)", string)


class NoisyTokenModel(llp.Model):
    """
    Simple model of noisy observation (by token).
    Samples continuations of `prompt` conditioning on each successive
    token being similar to the corresponding token from `string` input, using
    similarity function `is_similar`.
    """

    def __init__(self, prompt, string, is_similar=is_similar_levenshtein(1)):
        super().__init__()
        self.s = prompt
        self.ctx = self.new_context(self.s)
        self.remaining_tokens = self.llama.tokenize(string)
        self.is_similar = is_similar

    def step(self):
        # Actual next token
        true_token = self.remaining_tokens.pop(0)

        # Sample noisy version that is similar
        sampled_token = self.sample(
            llp.Transformer(self.ctx), proposal=self.proposal(true_token)
        )

        # Condition on constraint (not necessary if proposal is correct)
        # self.condition(self.is_similar(true_token, sampled_token))

        # Update generated string
        self.s += sampled_token

        # Check if done
        if len(self.remaining_tokens) == 0:
            self.observe(llp.Transformer(self.ctx), llp.EOS)
            self.finish()
            return

    def proposal(self, true_token):
        lm_logits = self.ctx.logits()
        mask = np.array(
            [0.0 if self.is_similar(true_token, v) else -np.inf for v in self.vocab()]
        )
        logprobs = llp.lognormalize(lm_logits + mask)
        return llp.TokenCategorical(logprobs)


class NoisyWordModel(llp.Model):
    """
    Noisy observations of whole words rather than single tokens.
    TODO: allow noisy word to be different number of tokens than observation.
    """

    def __init__(self, prompt, string, is_similar):
        super().__init__()
        self.s = prompt
        self.ctx = self.new_context(self.s)
        self.remaining_words = split_before_whitespace(string)
        self.is_similar = is_similar

    def step(self):
        true_word = self.remaining_words.pop(0)
        true_tokens = self.llama.tokenize(true_word)

        for i, tok in enumerate(true_tokens):
            self.s += self.sample(
                llp.Transformer(self.ctx), proposal=self.proposal(tok)
            )

        if len(self.remaining_words) == 0:
            self.observe(llp.Transformer(self.ctx), llp.EOS)
            self.finish()
            return

    def proposal(self, true_token):
        lm_logits = self.ctx.logits()
        mask = np.array(
            [0.0 if self.is_similar(true_token, v) else -np.inf for v in self.vocab()]
        )
        logprobs = llp.lognormalize(lm_logits + mask)
        return llp.TokenCategorical(logprobs)


class NoisyContextModel(llp.Model):
    """NB: Draft. Not correctly implemented yet."""

    def __init__(self, prompt, string, is_similar=is_similar_levenshtein(1)):
        super().__init__()
        self.s = prompt
        self.context = self.new_context(self.s)

    def step(self):
        # Noise context
        noised_s = self.add_deletion_noise(self.s)
        # print(f"{noised_s = }")
        self.s = noised_s
        self.context = self.new_context(noised_s)

        # Sample next token
        token = self.sample(llp.Transformer(self.context))

        if token == llp.EOS:
            self.finish()
            return

        self.s += token

    def add_deletion_noise(self, string):
        words = string.split()
        if len(words) > 1:
            words = [
                w
                for i, w in enumerate(words)
                if not self.maybe_drop(len(words) - i - 1)
            ]
            string = " ".join(words)
        return string

    def maybe_drop(self, distance):
        p = 0.95**distance
        return self.sample(llp.Categorical([p, 1 - p])) == 1


if __name__ == "__main__":
    # Create the model
    llp.LLaMAConfig.set_model_path(input("Path to GGML LLaMA model weights: "))
    model = NoisyTokenModel("", "ello, how art yo going today?")
    # Run SMC
    for i, p in enumerate(llp.smc_steer(model, 5, 15, verbose=False)):
        print(f"Particle {i}: {p} (weight {p.weight})")
else:
    llp.LLaMAConfig.set_model_path(llp.WEIGHTS_PATH)
