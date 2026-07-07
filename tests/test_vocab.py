from collections import Counter

from lbm.data.vocab import Vocab, time_gap_bucket


def test_vocab_reserves_pad_and_unk():
    vocab = Vocab.from_counts(Counter({"a": 3, "b": 1}))
    assert vocab.encode("<pad>") == 0
    assert vocab.token2id["<unk>"] == 1


def test_vocab_encodes_known_tokens_by_frequency():
    vocab = Vocab.from_counts(Counter({"a": 3, "b": 1}))
    assert vocab.encode("a") == 2
    assert vocab.encode("b") == 3


def test_vocab_encodes_unknown_token_as_unk():
    vocab = Vocab.from_counts(Counter({"a": 3}))
    assert vocab.encode("never_seen") == vocab.encode("<unk>")


def test_vocab_respects_top_k():
    vocab = Vocab.from_counts(Counter({"a": 3, "b": 2, "c": 1}), top_k=1)
    assert len(vocab) == 3  # pad + unk + top-1 token
    assert vocab.encode("c") == vocab.encode("<unk>")


def test_time_gap_bucket_boundaries():
    assert time_gap_bucket(5) == 1
    assert time_gap_bucket(30) == 2
    assert time_gap_bucket(120) == 3
    assert time_gap_bucket(1800) == 4
    assert time_gap_bucket(7200) == 5
