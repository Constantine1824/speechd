import numpy as np
import pytest

from speechsep.pipeline.cluster import cluster_speakers, estimate_num_speakers
from speechsep.schemas import Segment


def _two_clusters(n_per=3, dim=192, seed=0):
    """Build embeddings forming two well-separated cosine clusters."""
    rng = np.random.default_rng(seed)
    a = np.zeros(dim)
    a[0] = 1.0
    b = np.zeros(dim)
    b[1] = 1.0
    group_a = [a + 0.01 * rng.standard_normal(dim) for _ in range(n_per)]
    group_b = [b + 0.01 * rng.standard_normal(dim) for _ in range(n_per)]
    return group_a, group_b


def _segments(count):
    return [
        Segment(start=float(i), end=float(i) + 0.8, audio=np.zeros(16000))
        for i in range(count)
    ]


def test_estimate_num_speakers_finds_two():
    a, b = _two_clusters()
    embeddings = np.stack(a + b)
    assert estimate_num_speakers(embeddings, max_speakers=8) == 2


def test_cluster_speakers_separates_two_groups():
    a, b = _two_clusters()
    embeddings = a + b
    segs = _segments(len(embeddings))
    labeled = cluster_speakers(embeddings, segs, num_speakers=2)

    assert len(labeled) == len(embeddings)
    # First three came from group A, last three from group B.
    first_group = {labeled[i].speaker_id for i in range(3)}
    second_group = {labeled[i].speaker_id for i in range(3, 6)}
    assert len(first_group) == 1
    assert len(second_group) == 1
    assert first_group != second_group


def test_cluster_speakers_sorted_by_start_time():
    a, b = _two_clusters()
    embeddings = a + b
    segs = _segments(len(embeddings))
    # Shuffle so input is not already ordered.
    segs = segs[::-1]
    labeled = cluster_speakers(embeddings, segs, num_speakers=2)
    starts = [ls.start for ls in labeled]
    assert starts == sorted(starts)


def test_cluster_speakers_single_segment():
    emb = [np.ones(192) / np.sqrt(192)]
    segs = _segments(1)
    labeled = cluster_speakers(emb, segs)
    assert len(labeled) == 1
    assert labeled[0].speaker_id == 0


def test_cluster_speakers_length_mismatch_raises():
    a, _ = _two_clusters()
    with pytest.raises(ValueError):
        cluster_speakers(a, _segments(len(a) + 1), num_speakers=2)


def test_cluster_speakers_auto_estimates_when_k_none():
    a, b = _two_clusters()
    embeddings = a + b
    segs = _segments(len(embeddings))
    labeled = cluster_speakers(embeddings, segs, num_speakers=None, max_speakers=8)
    assert len({ls.speaker_id for ls in labeled}) == 2
