from collections import Counter

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


class Vocab:
    """Maps hashable tokens to contiguous integer ids. Reserves id 0 for
    padding and id 1 for unknown tokens, so every embedding table built
    from a Vocab can use id 0 as its padding_idx."""

    def __init__(self, tokens: list):
        self.token2id = {PAD_TOKEN: 0, UNK_TOKEN: 1}
        for token in tokens:
            self.token2id[token] = len(self.token2id)
        self.id2token = {i: t for t, i in self.token2id.items()}

    def __len__(self) -> int:
        return len(self.token2id)

    def encode(self, token) -> int:
        return self.token2id.get(token, self.token2id[UNK_TOKEN])

    @classmethod
    def from_counts(cls, counter: Counter, top_k: int | None = None) -> "Vocab":
        most_common = counter.most_common(top_k)
        tokens = [token for token, _ in most_common]
        return cls(tokens)


def time_gap_bucket(seconds: float) -> int:
    """Bucket a time gap (in seconds) since the previous event in a
    session into a discrete id in [1, 5]. Id 0 is reserved for padding,
    matching the convention used by Vocab-encoded fields."""
    if seconds < 10:
        return 1
    if seconds < 60:
        return 2
    if seconds < 300:
        return 3
    if seconds < 3600:
        return 4
    return 5
