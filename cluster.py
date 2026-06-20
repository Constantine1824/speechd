import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

from schemas import LabeledSegment, Segment


def _cluster(embeddings: np.ndarray, k: int) -> np.ndarray:
    """Run spectral clustering with cosine affinity for a fixed k."""
    return SpectralClustering(
        n_clusters=k,
        affinity="cosine",
        random_state=42,
    ).fit_predict(embeddings)


def _best_clustering(
    embeddings: np.ndarray,
    max_speakers: int = 8,
) -> tuple[int, np.ndarray]:
    """
    Find the speaker count with the highest silhouette score and return
    both the chosen k and the labels for that k — so callers don't have to
    re-cluster.

    Tries k = 2..min(max_speakers, n-1) and picks the best silhouette score.
    """
    n = len(embeddings)
    if n <= 2:
        # One cluster per segment (or a single cluster for n==1).
        return min(n, 2), np.arange(n) if n <= 2 else np.zeros(n, dtype=int)

    best_k, best_score, best_labels = 2, -1.0, None
    upper = min(max_speakers, n - 1)

    for k in range(2, upper + 1):
        labels = _cluster(embeddings, k)
        score = silhouette_score(embeddings, labels, metric="cosine")
        if score > best_score:
            best_score, best_k, best_labels = score, k, labels

    print(f"Estimated {best_k} speaker(s) (silhouette={best_score:.3f})")
    return best_k, best_labels


def estimate_num_speakers(
    embeddings: np.ndarray,
    max_speakers: int = 8,
) -> int:
    """
    Estimate the number of speakers using silhouette score over a range
    of cluster counts. Falls back to k=2 if only 1 or 2 segments exist.

    Parameters
    ----------
    embeddings   : (N, D) array of speaker embeddings
    max_speakers : upper bound on number of speakers to try

    Returns
    -------
    Estimated number of speakers
    """
    return _best_clustering(embeddings, max_speakers=max_speakers)[0]


def cluster_speakers(
    embeddings: list[np.ndarray],
    segments: list[Segment],
    num_speakers: int | None = None,
    max_speakers: int = 8,
) -> list[LabeledSegment]:
    """
    Assign a speaker label to each segment via spectral clustering.

    Parameters
    ----------
    embeddings   : list of (192,) speaker embedding arrays (from embed.py)
    segments     : corresponding list of Segment objects (from vad.py)
    num_speakers : if known, provide it directly; otherwise estimated automatically
    max_speakers : upper bound when auto-estimating

    Returns
    -------
    List of LabeledSegment objects sorted by start time
    """
    if len(embeddings) != len(segments):
        raise ValueError(
            f"embeddings ({len(embeddings)}) and segments ({len(segments)}) must match"
        )

    # Stack and L2-normalize embeddings
    emb_matrix = normalize(np.stack(embeddings), norm="l2")  # (N, 192)

    # Handle degenerate case: single segment
    if len(embeddings) == 1:
        return [LabeledSegment(segment=segments[0], speaker_id=0)]

    # Estimate k (reusing its labels) or cluster directly for a known k.
    if num_speakers:
        labels = _cluster(emb_matrix, num_speakers)
    else:
        _, labels = _best_clustering(emb_matrix, max_speakers=max_speakers)

    labeled = [
        LabeledSegment(segment=seg, speaker_id=int(label))
        for seg, label in zip(segments, labels)
    ]

    # Sort by start time
    labeled.sort(key=lambda x: x.start)

    # Summary
    unique_speakers = set(ls.speaker_id for ls in labeled)
    print(
        f"[cluster] Assigned {len(labeled)} segments to {len(unique_speakers)} speaker(s)"
    )

    return labeled
