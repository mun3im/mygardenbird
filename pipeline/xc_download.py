"""Shared Xeno-Canto download helpers for Stage 2 and Stage 2a.

Both download stages probe audio, convert MP3 to mono FLAC and resolve an API
key in exactly the same way. They used to carry their own copies: the probe was
byte-identical in both files and the converter differed only in formatting, so
a fix to one silently left the other wrong. The duration-gate bug of
2026-11-07 had to be patched twice for that reason, and the API key lookup had
drifted far enough apart that Stage 2 raised a bare 401 under an environment
where Stage 1 worked.

Resampling is deliberately not done here -- it belongs to Stage 6, so these
helpers preserve the source sample rate and assert that ffmpeg did too.
"""

import os
import subprocess

MIN_DURATION_S = 3.0   # recordings shorter than this are skipped at download


def resolve_api_key(cli_key=None, script_dir=None, announce=True):
    """Return an XC API key from CLI arg, XENO_API_KEY, then xc_key.txt.

    All three download/metadata stages must agree on this order, or a machine
    that can fetch metadata can still fail to download with an unexplained
    401.

    Returns None when no key is configured; API v3 allows some unauthenticated
    use, so callers decide whether that is fatal.
    """
    if cli_key:
        return cli_key

    env_key = os.environ.get("XENO_API_KEY")
    if env_key:
        if announce:
            print("Loaded API key from XENO_API_KEY\n")
        return env_key.strip()

    script_dir = script_dir or os.path.dirname(os.path.abspath(__file__))
    key_file = os.path.join(script_dir, "xc_key.txt")
    if os.path.exists(key_file):
        with open(key_file) as f:
            key = f.read().strip()
        if key:
            if announce:
                print(f"Loaded API key from {key_file}\n")
            return key
    return None


def is_nd_licence(licence):
    """True when a licence URL carries the CC NoDerivatives term.

    ND forbids derivative works and 3-second segmentation is a derivative, so
    these recordings cannot be redistributed in the dataset. Matching is on the
    '-nd' token rather than a fixed list of URLs, because Xeno-Canto carries
    ND under several CC versions (2.5, 3.0 and 4.0 all appear in this corpus).
    """
    u = (licence or "").strip().lower().rstrip("/")
    return u.endswith("-nd") or "-nd/" in u + "/"


def probe_audio(path):
    """Return (sample_rate, duration_s) of an audio file via ffprobe.

    Either value is None if it cannot be determined, which callers must treat
    as a failure rather than a pass: ffprobe writes nothing to stdout when the
    input is not decodable audio at all, and a download that returned an HTML
    error body instead of a recording lands here looking like a missing field.

    Duration is read from the stream, falling back to the container, because
    the stream header can lack it in a stream-copied or truncated file.
    """
    def _run(args):
        try:
            r = subprocess.run(["ffprobe", "-v", "error", *args, "-of",
                                "csv=p=0", path],
                               capture_output=True, text=True, timeout=15)
            return r.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return ""

    sr = dur = None
    parts = _run(["-select_streams", "a:0",
                  "-show_entries", "stream=sample_rate,duration"]).split(",")
    try:
        if len(parts) > 0 and parts[0]:
            sr = int(parts[0])
        if len(parts) > 1 and parts[1]:
            dur = float(parts[1])
    except ValueError:
        pass

    if dur is None:
        # N/A or absent in the stream header; the container usually still knows.
        text = _run(["-show_entries", "format=duration"])
        try:
            if text and text != "N/A":
                dur = float(text)
        except ValueError:
            pass

    return sr, dur


def mp3_to_mono_flac(mp3_path, flac_path, min_duration_s=MIN_DURATION_S):
    """Convert an MP3 to mono FLAC, skipping anything too short or unreadable.

    Pre-conversion:  recordings shorter than `min_duration_s` are skipped, as
                     is input whose duration cannot be determined at all.
    Post-conversion: asserts the output preserves the source sample rate.
                     Resampling is intentionally deferred to Stage 6.

    Returns True if the file was converted and passes all checks, else False
    (deleting any partial output).
    """
    src_sr, src_dur = probe_audio(mp3_path)

    # An unknown duration is a skip, not a pass: ffprobe reports nothing for
    # input that is not decodable audio, so the permissive reading of None
    # here let short files and HTML error bodies saved as .mp3 reach ffmpeg.
    name = os.path.basename(mp3_path)
    if src_dur is None:
        print(f"  Skipping XC{name}: duration unknown (not decodable audio?)")
        return False
    if src_dur < min_duration_s:
        print(f"  Skipping XC{name}: {src_dur:.2f}s < {min_duration_s}s")
        return False

    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", mp3_path,
                "-ac", "1",      # mono — channel change only, no -ar flag
                "-c:a", "flac",
                flac_path,
            ],
            capture_output=True, timeout=120,
        )
        if not (os.path.isfile(flac_path) and os.path.getsize(flac_path) > 0):
            return False

        # Assert sample rate was preserved
        dst_sr, _ = probe_audio(flac_path)
        if src_sr is not None and dst_sr is not None and src_sr != dst_sr:
            print(f"  ERROR: sample rate changed {src_sr} Hz -> {dst_sr} Hz "
                  f"in {flac_path}")
            print("  Deleting corrupted output.")
            os.unlink(flac_path)
            return False
        return True

    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"  ffmpeg error: {e}")
        if os.path.isfile(flac_path):
            os.unlink(flac_path)
        return False
