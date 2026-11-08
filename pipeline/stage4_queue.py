#!/usr/bin/env python3
"""
stage4_queue.py — Folder scanning and annotation-status model for the Stage 4
annotators.

Status is derived from the `.txt` sidecar beside each audio file, following the
convention already in use:

    no sidecar        → TO_EVALUATE  (not yet looked at)
    zero-length file  → REJECTED     (listened to, deliberately not used)
    non-empty file    → ACCEPTED     (one or more segments saved)

GUI-free so it can be unit-tested and reused.
"""

import os
from enum import Enum

AUDIO_EXTENSIONS = (".flac", ".wav", ".mp3", ".m4a", ".ogg", ".aac", ".wma")


# Keycap emoji 1-10. An accepted file always has between 1 and MAX_SEGMENTS
# (10) segments, so the count itself can stand in for a generic tick.
COUNT_EMOJI = {
    1: "1️⃣", 2: "2️⃣", 3: "3️⃣", 4: "4️⃣", 5: "5️⃣",
    6: "6️⃣", 7: "7️⃣", 8: "8️⃣", 9: "9️⃣", 10: "🔟",
}


class Status(Enum):
    TO_EVALUATE = "to_evaluate"
    REJECTED = "rejected"
    ACCEPTED = "accepted"

    @property
    def emoji(self):
        return {
            Status.TO_EVALUATE: "⬜",
            Status.REJECTED: "❌",
            Status.ACCEPTED: "✅",
        }[self]

    @property
    def label(self):
        return {
            Status.TO_EVALUATE: "To evaluate",
            Status.REJECTED: "Rejected",
            Status.ACCEPTED: "Accepted",
        }[self]


def annotation_path_for(audio_path):
    """Sidecar path for an audio file (mirrors stage4_core)."""
    return os.path.splitext(audio_path)[0] + ".txt"


def status_for(audio_path):
    """Classify a single audio file from its sidecar."""
    txt = annotation_path_for(audio_path)
    try:
        if os.path.getsize(txt) == 0:
            return Status.REJECTED
        return Status.ACCEPTED
    except OSError:
        # Missing sidecar (and unreadable ones) count as not yet evaluated.
        return Status.TO_EVALUATE


def segment_count(audio_path):
    """Number of saved segments, or 0 for rejected/unevaluated files."""
    txt = annotation_path_for(audio_path)
    try:
        with open(txt) as f:
            return sum(1 for line in f
                       if line.strip() and not line.startswith("#"))
    except OSError:
        return 0


class QueueEntry:
    __slots__ = ("path", "status", "n_segments")

    def __init__(self, path):
        self.path = path
        self.refresh()

    def refresh(self):
        self.status = status_for(self.path)
        self.n_segments = segment_count(self.path)

    @property
    def name(self):
        return os.path.basename(self.path)

    @property
    def marker(self):
        """Leading glyph: the segment count for accepted files, else status."""
        if self.status is Status.ACCEPTED:
            # Falls back to the tick if a file somehow holds more segments
            # than the cap (e.g. a hand-edited .txt).
            return COUNT_EMOJI.get(self.n_segments, Status.ACCEPTED.emoji)
        return self.status.emoji

    @property
    def display(self):
        """Row text for the file list, e.g. '4️⃣ xc12345.flac'."""
        return f"{self.marker} {self.name}"

    def __repr__(self):
        return f"<QueueEntry {self.status.value} {self.name}>"


def scan_folder(folder, recursive=True, extensions=AUDIO_EXTENSIONS):
    """Return QueueEntry objects for audio files under `folder`, sorted by name.

    Sorting is by (directory, filename) so a recursive scan stays grouped by
    subfolder (e.g. grade A/B/C/D/E) rather than interleaving them.
    """
    found = []
    if recursive:
        for root, _dirs, files in os.walk(folder):
            for fn in files:
                if fn.lower().endswith(extensions):
                    found.append(os.path.join(root, fn))
    else:
        try:
            for fn in os.listdir(folder):
                p = os.path.join(folder, fn)
                if os.path.isfile(p) and fn.lower().endswith(extensions):
                    found.append(p)
        except OSError:
            return []

    found.sort(key=lambda p: (os.path.dirname(p).lower(), os.path.basename(p).lower()))
    return [QueueEntry(p) for p in found]


def summarise(entries):
    """Count entries per status."""
    counts = {s: 0 for s in Status}
    for e in entries:
        counts[e.status] += 1
    return counts


def format_summary(entries):
    """One-line progress summary for a status bar."""
    c = summarise(entries)
    total = len(entries)
    done = c[Status.ACCEPTED] + c[Status.REJECTED]
    return (f"{done}/{total} done  "
            f"{Status.ACCEPTED.emoji} {c[Status.ACCEPTED]}  "
            f"{Status.REJECTED.emoji} {c[Status.REJECTED]}  "
            f"{Status.TO_EVALUATE.emoji} {c[Status.TO_EVALUATE]}")


def next_unevaluated(entries, from_index):
    """Index of the next TO_EVALUATE entry after `from_index`, else None.

    Used by 'save and advance' so the queue skips files already dealt with.
    """
    for i in range(from_index + 1, len(entries)):
        if entries[i].status is Status.TO_EVALUATE:
            return i
    return None
