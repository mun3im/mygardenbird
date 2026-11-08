#!/usr/bin/env python3
"""
Stage4b_annotate_qt.py — BirdClip Picker.

Chooses which 3-second slices of a bird recording become dataset clips.

A faster alternative to Stage4_annotate_segments.py for working through large
backlogs: pick a folder once, then move file-to-file from the keyboard. Both
tools share stage4_core, so detection, SNR and the .txt output are identical.

Status marker in the file list:
    ⬜       no sidecar     — still to evaluate
    ❌       zero-length    — rejected
    1️⃣ … 🔟  has segments   — accepted, showing the segment count

Run:  python3 Stage4b_annotate_qt.py [--folder DIR] [--file FILE]
"""

import argparse
import json
import os
import signal
import sys
import traceback
from datetime import datetime

import numpy as np
import librosa
import sounddevice as sd

from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

from stage4_core import (
    FREQ_MIN, DEFAULT_FREQ_CUTOFF, MP3_NYQUIST_VALUES,
    DISPLAY_HOP, SEGMENT_DURATION, MAX_SEGMENTS, INITIAL_THRESHOLD_MULT,
    DEFAULT_SOUND_DIR, get_profile,
    lookup_xc_type, calculate_enhanced_snr, compute_spectrograms,
    detect_sound_blobs, compute_freq_bounds, create_fixed_segments,
    annotation_path_for, load_annotation,
)
from stage4_queue import Status, scan_folder, format_summary, next_unevaluated

pg.setConfigOptions(imageAxisOrder="row-major", antialias=False)

APP_NAME = "BirdClip Picker"

STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          ".stage4b_state.json")

SELECTED_PEN = pg.mkPen((140, 255, 140), width=3)
SELECTED_BRUSH = pg.mkBrush(140, 255, 140, 70)
NORMAL_PEN = pg.mkPen((255, 230, 60), width=2)
NORMAL_BRUSH = pg.mkBrush(255, 230, 60, 45)

# Segment-number badges sit on a dark plasma spectrogram, so they use opaque
# fills and white text rather than the translucent box brushes.
LABEL_TEXT_COLOR = (255, 255, 255)
LABEL_FILL_SELECTED = pg.mkBrush(0, 130, 0, 235)
LABEL_FILL_NORMAL = pg.mkBrush(15, 15, 15, 215)
LABEL_POINT_SIZE = 22          # roughly double the previous default

# Waveform spans. The waveform sits on white, so the spectrogram's bright
# yellow/green washes out there and darker, more saturated fills are used.
# Amplitude has no band to bound the way frequency does, so these span the
# full height.
WAVE_SPAN_SELECTED = pg.mkBrush(0, 150, 0, 60)
WAVE_SPAN_NORMAL = pg.mkBrush(205, 125, 0, 30)
WAVE_EDGE_SELECTED = pg.mkPen((0, 115, 0), width=2)
WAVE_EDGE_NORMAL = pg.mkPen((205, 140, 20), width=1)


class ScrollPanViewBox(pg.ViewBox):
    """ViewBox where the wheel scrolls through time and Shift+wheel zooms.

    pyqtgraph's default is the reverse. Scrolling is the more common action
    when walking a long recording, so it gets the unmodified wheel.
    """

    def wheelEvent(self, ev, axis=None):
        if ev.modifiers() & QtCore.Qt.ShiftModifier:
            super().wheelEvent(ev, axis)
            return

        (x_lo, x_hi), _ = self.viewRange()
        span = x_hi - x_lo
        # delta() is in eighths of a degree; one notch is 120.
        notches = ev.delta() / 120.0
        shift = -notches * span * 0.15

        x_min, x_max = self._pan_bounds()
        if x_min is not None:
            if shift < 0:
                shift = max(shift, x_min - x_lo)
            else:
                shift = min(shift, x_max - x_hi)

        if shift:
            self.setXRange(x_lo + shift, x_hi + shift, padding=0)
        ev.accept()

    def _pan_bounds(self):
        limits = self.state.get("limits", {}).get("xLimits", [None, None])
        return limits[0], limits[1]


def _label_font():
    """Bold, large font for the segment-number badges."""
    font = QtGui.QFont()
    font.setPointSize(LABEL_POINT_SIZE)
    font.setBold(True)
    return font


def _mono_font(size=11):
    """Monospace font that exists on macOS, Windows and Linux alike."""
    try:
        font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
        font.setPointSize(size)
        return font
    except Exception:                                 # noqa: BLE001
        return QtGui.QFont("monospace", size)


class SegmentBox(pg.GraphicsObject):
    """A fixed-width segment drawn tight to its detected frequency band.

    pyqtgraph's LinearRegionItem always spans the full y-axis, which hides the
    spectrogram and gives no sense of the segment's frequency extent, so the
    box is drawn explicitly from the segment's (fmin, fmax) bounds instead.
    """

    def __init__(self, index, start, fmin, fmax, selected, owner):
        super().__init__()
        self.index = index
        self.start = start
        self.fmin, self.fmax = fmin, fmax
        self.selected = selected
        self.owner = owner
        self._drag_from = None
        self.setZValue(10)
        self.setAcceptHoverEvents(True)
        self._rebuild()

    def _rebuild(self):
        self._rect = QtCore.QRectF(self.start, self.fmin,
                                   SEGMENT_DURATION, self.fmax - self.fmin)
        self.prepareGeometryChange()
        self.update()

    def boundingRect(self):
        return self._rect

    def paint(self, painter, *_args):
        painter.setPen(SELECTED_PEN if self.selected else NORMAL_PEN)
        painter.setBrush(SELECTED_BRUSH if self.selected else NORMAL_BRUSH)
        painter.drawRect(self._rect)

    def set_start(self, start):
        self.start = start
        self._rebuild()

    def set_selected(self, selected):
        if self.selected != selected:
            self.selected = selected
            self.update()

    # Dragging moves the box in time only; width and frequency band are fixed.
    def mousePressEvent(self, ev):
        if ev.button() != QtCore.Qt.LeftButton:
            ev.ignore()
            return
        ev.accept()
        self._drag_from = ev.pos().x() - self.start
        self.owner.select_segment(self.index)

    def mouseMoveEvent(self, ev):
        if self._drag_from is None:
            ev.ignore()
            return
        ev.accept()
        self.owner.preview_segment_move(self.index, ev.pos().x() - self._drag_from)

    def mouseReleaseEvent(self, ev):
        if self._drag_from is None:
            ev.ignore()
            return
        ev.accept()
        self._drag_from = None
        self.owner.commit_segment_move(self.index, self.start)


class WaveSpan(pg.LinearRegionItem):
    """Read-only time span marking a segment on the waveform.

    Unlike SegmentBox this spans the full height, which is what's wanted here:
    the y-axis is amplitude, so there is no band to draw tight to. Editing
    stays on the spectrogram, so the span is immovable and only reports clicks
    as a selection.
    """

    def __init__(self, index, start, selected, owner):
        super().__init__(values=(start, start + SEGMENT_DURATION),
                         movable=False)
        self.index = index
        self.owner = owner
        self.setZValue(-5)          # behind the trace, not over it
        self.set_selected(selected)

    def set_start(self, start):
        self.setRegion((start, start + SEGMENT_DURATION))

    def set_selected(self, selected):
        self.selected = selected
        brush = WAVE_SPAN_SELECTED if selected else WAVE_SPAN_NORMAL
        # setBrush() assigns currentBrush itself, but hoverBrush keeps the
        # default translucent blue, which would flash on hover.
        self.setBrush(brush)
        self.hoverBrush = brush
        edge = WAVE_EDGE_SELECTED if selected else WAVE_EDGE_NORMAL
        for line in self.lines:
            line.setPen(edge)
        self.update()

    def mouseClickEvent(self, ev):
        if ev.button() == QtCore.Qt.LeftButton:
            ev.accept()
            self.owner.select_segment(self.index)
        else:
            ev.ignore()

    # A non-movable LinearRegionItem still accepts drags; swallow them so a
    # stray drag on the waveform doesn't look like a failed edit.
    def mouseDragEvent(self, ev):
        ev.ignore()


# ── Background loading ───────────────────────────────────────────────────────
class LoadWorker(QtCore.QObject):
    """Loads audio + spectrograms + detection off the GUI thread."""
    finished = QtCore.Signal(object, int)
    failed = QtCore.Signal(str, str, int)

    def __init__(self, path, thresh, erode, dilate, generation=0):
        super().__init__()
        self.path, self.thresh = path, thresh
        self.erode, self.dilate = erode, dilate
        self.generation = generation

    @QtCore.Slot()
    def run(self):
        try:
            y, sr = librosa.load(self.path, sr=None)
            duration = len(y) / sr
            if duration < SEGMENT_DURATION:
                self.failed.emit(self.path,
                                 f"too short ({duration:.2f}s < {SEGMENT_DURATION}s)",
                                 self.generation)
                return
            S_det, S_db_full, freqs_full = compute_spectrograms(y, sr)
            events, _mask = detect_sound_blobs(
                S_det, sr, threshold_mult=self.thresh,
                erode_cycles=self.erode, dilate_cycles=self.dilate)
            segs, bounds = create_fixed_segments(
                events, duration, S_db_full=S_db_full,
                freqs_full=freqs_full, sr=sr)
            self.finished.emit({
                "path": self.path, "y": y, "sr": sr, "duration": duration,
                "S_det": S_det, "S_db_full": S_db_full, "freqs_full": freqs_full,
                "segments": segs, "bounds": bounds,
            }, self.generation)
        except Exception as exc:                      # noqa: BLE001
            self.failed.emit(self.path, str(exc), self.generation)


class DetectWorker(QtCore.QObject):
    """Re-runs blob detection for already-loaded audio."""
    finished = QtCore.Signal(object, int)

    def __init__(self, S_det, S_db_full, freqs_full, sr, duration,
                 thresh, erode, dilate, generation=0):
        super().__init__()
        self.S_det, self.S_db_full = S_det, S_db_full
        self.freqs_full, self.sr, self.duration = freqs_full, sr, duration
        self.thresh, self.erode, self.dilate = thresh, erode, dilate
        self.generation = generation

    @QtCore.Slot()
    def run(self):
        events, _mask = detect_sound_blobs(
            self.S_det, self.sr, threshold_mult=self.thresh,
            erode_cycles=self.erode, dilate_cycles=self.dilate)
        segs, bounds = create_fixed_segments(
            events, self.duration, S_db_full=self.S_db_full,
            freqs_full=self.freqs_full, sr=self.sr)
        self.finished.emit((segs, bounds), self.generation)


# ── Main window ──────────────────────────────────────────────────────────────
class Annotator(QtWidgets.QMainWindow):
    def __init__(self, folder=None, single_file=None):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(1500, 900)

        self.entries = []
        self.index = -1
        self.root_folder = None         # folder chosen by the user, for restore
        self.audio = None               # dict from LoadWorker
        self.segments = []
        self.bounds = []
        self.selected = None
        self.dirty = False              # unsaved hand edits
        self.region_items = []
        self.label_items = []
        self.wave_items = []
        self._thread = None
        self._worker = None
        self._dthread = None
        self._dworker = None
        self._retired_threads = []
        self._load_generation = 0
        self._loading = False

        self._build_ui()
        self._install_shortcuts()
        if not self._restore_geometry():
            self._centre_on_primary()

        self.play_timer = QtCore.QTimer(self)
        self.play_timer.setInterval(40)
        self.play_timer.timeout.connect(self._tick_playback)
        self.play_span = None

        if single_file:
            self.root_folder = os.path.dirname(single_file)
            self.entries = scan_folder(self.root_folder, recursive=False)
            match = [i for i, e in enumerate(self.entries)
                     if os.path.abspath(e.path) == os.path.abspath(single_file)]
            self._refresh_list()
            self.load_index(match[0] if match else 0)
        else:
            # Reopen the folder from the previous session unless one was given.
            start = folder or self._last_dir()
            if start and os.path.isdir(start):
                self.open_folder(start)
            else:
                self.statusBar().showMessage(
                    "Open a folder to begin (last folder is remembered).")

    # ── UI construction ──────────────────────────────────────────────────
    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)

        # Left: file queue
        left = QtWidgets.QVBoxLayout()
        btn_open = QtWidgets.QPushButton("Open folder…")
        btn_open.clicked.connect(self.choose_folder)
        left.addWidget(btn_open)

        self.filter_box = QtWidgets.QComboBox()
        self.filter_box.addItems(["All files", "⬜ To evaluate",
                                  "✅ Accepted", "❌ Rejected"])
        self.filter_box.currentIndexChanged.connect(self._refresh_list)
        left.addWidget(self.filter_box)

        self.list_widget = QtWidgets.QListWidget()
        self.list_widget.setFont(_mono_font())
        self.list_widget.itemSelectionChanged.connect(self._on_list_selection)
        left.addWidget(self.list_widget, 1)

        self.summary_label = QtWidgets.QLabel("No folder loaded")
        self.summary_label.setWordWrap(True)
        left.addWidget(self.summary_label)

        left_panel = QtWidgets.QWidget()
        left_panel.setLayout(left)
        left_panel.setFixedWidth(330)
        root.addWidget(left_panel)

        # Right: plots + controls
        right = QtWidgets.QVBoxLayout()

        self.spec_plot = pg.PlotWidget(viewBox=ScrollPanViewBox())
        self.spec_plot.setLabel("left", "Frequency (Hz)")
        self.spec_plot.setMouseEnabled(x=True, y=False)
        self.spec_img = pg.ImageItem()
        self.spec_plot.addItem(self.spec_img)
        self.spec_cursor = pg.InfiniteLine(angle=90, pen=pg.mkPen("r", width=2))
        self.spec_cursor.setVisible(False)
        self.spec_plot.addItem(self.spec_cursor)
        right.addWidget(self.spec_plot, 3)

        self.wave_plot = pg.PlotWidget(viewBox=ScrollPanViewBox())
        self.wave_plot.setLabel("left", "Amplitude")
        self.wave_plot.setLabel("bottom", "Time (s)")
        self.wave_plot.setMouseEnabled(x=True, y=False)
        self.wave_plot.setBackground("w")
        self.wave_curve = self.wave_plot.plot(pen=pg.mkPen((10, 90, 170), width=1))
        self.wave_plot.showGrid(x=True, y=False, alpha=0.25)
        self.wave_cursor = pg.InfiniteLine(angle=90, pen=pg.mkPen("r", width=2))
        self.wave_cursor.setVisible(False)
        self.wave_plot.addItem(self.wave_cursor)
        self.wave_plot.setXLink(self.spec_plot)
        right.addWidget(self.wave_plot, 1)

        # Detection controls
        ctrl = QtWidgets.QHBoxLayout()
        self.sl_thresh = self._slider(100, 600, int(INITIAL_THRESHOLD_MULT * 100), 25)
        self.sl_erode = self._slider(0, 4, 1, 1)
        self.sl_dilate = self._slider(0, 4, 1, 1)
        for lbl, sl in (("Threshold×median", self.sl_thresh),
                        ("Erode", self.sl_erode),
                        ("Dilate", self.sl_dilate)):
            box = QtWidgets.QVBoxLayout()
            box.addWidget(QtWidgets.QLabel(lbl))
            box.addWidget(sl)
            ctrl.addLayout(box)
            sl.setToolTip("Re-runs detection and replaces all segments "
                          "for this file — edits are discarded.")
            sl.sliderReleased.connect(self.rerun_detection)

        fbox = QtWidgets.QVBoxLayout()
        fbox.addWidget(QtWidgets.QLabel("Max freq (Hz)"))
        self.freq_combo = QtWidgets.QComboBox()
        self.freq_combo.currentIndexChanged.connect(self._apply_freq_cutoff)
        fbox.addWidget(self.freq_combo)
        ctrl.addLayout(fbox)
        right.addLayout(ctrl)

        # Action buttons
        actions = QtWidgets.QHBoxLayout()
        for text, slot in (("▶ Play (Space)", self.toggle_play),
                           ("Zoom seg (Z)", self.zoom_to_selected),
                           ("Full (F)", self.zoom_full),
                           ("Delete seg (Del)", self.delete_selected),
                           ("✅ Save + next (S)", self.save_and_next),
                           ("❌ Reject + next (R)", self.reject_and_next),
                           ("← Prev (P)", self.prev_file),
                           ("Next → (N)", self.next_file)):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(slot)
            actions.addWidget(b)
            if text.startswith("▶"):
                self.play_button = b
        right.addLayout(actions)

        self.info_label = QtWidgets.QLabel("Open a folder to begin.")
        self.info_label.setFont(_mono_font())
        right.addWidget(self.info_label)

        root.addLayout(right, 1)
        self.statusBar().showMessage("Ready")

    @staticmethod
    def _slider(lo, hi, val, step):
        s = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        s.setMinimum(lo); s.setMaximum(hi); s.setValue(val)
        s.setSingleStep(step); s.setPageStep(step)
        return s

    def _install_shortcuts(self):
        def bind(seq, fn):
            QtGui.QShortcut(QtGui.QKeySequence(seq), self, activated=fn)
        bind("Space", self.toggle_play)
        bind("S", self.save_and_next)
        bind("R", self.reject_and_next)
        bind("N", self.next_file)
        bind("P", self.prev_file)
        bind("Delete", self.delete_selected)
        bind("Backspace", self.delete_selected)
        bind("Left", lambda: self.step_selection(-1))
        bind("Right", lambda: self.step_selection(+1))
        # Tab/Shift+Tab cycle segments so Tab-Space-Tab-Space auditions them
        # one by one. Bound on the window, which takes precedence over focus
        # traversal, so Tab never moves focus to a slider or button instead.
        bind("Tab", lambda: self.step_selection(+1))
        bind("Shift+Tab", lambda: self.step_selection(-1))
        bind("Backtab", lambda: self.step_selection(-1))
        bind("Z", self.zoom_to_selected)
        bind("F", self.zoom_full)
        bind("+", lambda: self.zoom_by(0.5))
        bind("=", lambda: self.zoom_by(0.5))      # unshifted '+' on most layouts
        bind("-", lambda: self.zoom_by(2.0))
        for n in range(1, 10):
            bind(str(n), lambda n=n: self.select_segment(n - 1))
        bind("0", lambda: self.select_segment(9))

    # ── Folder / queue ───────────────────────────────────────────────────
    def choose_folder(self):
        start = self._last_dir() or DEFAULT_SOUND_DIR
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Choose folder of audio files", start)
        if folder:
            self.open_folder(folder)

    def open_folder(self, folder):
        self.entries = scan_folder(folder)
        # Remember the folder actually chosen. The scan is recursive, so the
        # first entry usually lives in a subfolder (e.g. '<species>/A') and
        # cannot be used to recover this path on quit.
        self.root_folder = folder
        self._save_last_dir(folder)
        if not self.entries:
            QtWidgets.QMessageBox.warning(self, "Empty",
                                          f"No audio files found under:\n{folder}")
            return
        self._refresh_list()
        start = next_unevaluated(self.entries, -1)
        self.load_index(start if start is not None else 0)

    def _visible_indices(self):
        mode = self.filter_box.currentIndex()
        wanted = {1: Status.TO_EVALUATE, 2: Status.ACCEPTED, 3: Status.REJECTED}.get(mode)
        return [i for i, e in enumerate(self.entries)
                if wanted is None or e.status is wanted]

    def _refresh_list(self):
        self.list_widget.blockSignals(True)
        self.list_widget.clear()
        for i in self._visible_indices():
            item = QtWidgets.QListWidgetItem(self.entries[i].display)
            item.setData(QtCore.Qt.UserRole, i)
            if i == self.index:
                item.setBackground(QtGui.QColor(220, 240, 255))
            self.list_widget.addItem(item)
        self.list_widget.blockSignals(False)
        self.summary_label.setText(format_summary(self.entries))

    def _on_list_selection(self):
        items = self.list_widget.selectedItems()
        if items:
            idx = items[0].data(QtCore.Qt.UserRole)
            if idx != self.index:
                self.load_index(idx)

    # ── Loading ──────────────────────────────────────────────────────────
    def load_index(self, idx):
        if not (0 <= idx < len(self.entries)):
            return
        if self.dirty and not self._confirm_discard():
            return
        self.stop_playback()
        # Cast any in-flight load loose rather than waiting on it; waiting
        # blocks the GUI thread for as long as the old file takes to finish.
        self._retire_thread("_thread")
        self._load_generation += 1
        generation = self._load_generation

        self.index = idx
        entry = self.entries[idx]
        # self.audio still holds the PREVIOUS file until this load lands, so a
        # separate flag is needed to stop save/reject acting on stale data.
        self._loading = True
        self.audio = None
        self.statusBar().showMessage(f"Loading {entry.name} …")
        self.info_label.setText(f"Loading {entry.name} …")

        self._thread = QtCore.QThread(self)
        self._worker = LoadWorker(entry.path, self.sl_thresh.value() / 100.0,
                                  self.sl_erode.value(), self.sl_dilate.value(),
                                  generation)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        # Queued + decorated slots are required: a bare lambda has no receiver
        # QObject, so Qt runs it in the WORKER thread, and touching widgets
        # there crashes on macOS ("NSWindow geometry should only be modified
        # on the main thread").
        self._worker.finished.connect(self._on_loaded,
                                      QtCore.Qt.QueuedConnection)
        self._worker.failed.connect(self._on_load_failed,
                                    QtCore.Qt.QueuedConnection)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.start()

    def _retire_thread(self, attr):
        """Stop tracking a worker thread without blocking the GUI.

        The workers run one long uninterruptible call (librosa load + STFT +
        blob detection, ~5s on a 700s file). QThread.quit() cannot interrupt
        that, so waiting on it froze the UI for the remainder — the beachball
        seen on save-and-next. Instead the old thread is cast loose: it is
        kept referenced so Qt does not destroy a running thread, it deletes
        itself when it finishes, and its result is ignored because the
        generation counter has moved on.
        """
        t = getattr(self, attr, None)
        if t is None:
            return
        try:
            if t.isRunning():
                # Hold a reference so Qt never destroys a running QThread, and
                # drop it once the thread signals that it has finished. The
                # connection is queued to this window, so the cleanup always
                # runs on the GUI thread.
                self._retired_threads.append(t)
                t.finished.connect(self._forget_thread,
                                   QtCore.Qt.QueuedConnection)
        except RuntimeError:
            pass              # already deleted by Qt
        setattr(self, attr, None)

    @QtCore.Slot()
    def _forget_thread(self):
        """Drop references to retired threads that have now finished."""
        still_running = []
        for t in self._retired_threads:
            try:
                (still_running if t.isRunning() else []).append(t)
            except RuntimeError:
                pass          # already deleted
        self._retired_threads = still_running

    def _wait_for_thread(self, attr, timeout_ms=15000):
        """Block until a worker finishes. Only used on shutdown."""
        t = getattr(self, attr, None)
        if t is not None:
            try:
                if t.isRunning():
                    t.quit()
                    t.wait(timeout_ms)
            except RuntimeError:
                pass
            setattr(self, attr, None)

    @QtCore.Slot(object, int)
    def _on_loaded(self, data, generation=None):
        # Ignore a result from a load the user has already navigated past.
        if generation is not None and generation != self._load_generation:
            return
        self.audio = data
        path = data["path"]

        ann = annotation_path_for(path)
        # load_annotation warns on a missing file; absent sidecars are the
        # normal case here, so only call it when one actually exists.
        saved = load_annotation(ann) if os.path.exists(ann) else []
        if saved:
            self.segments = list(saved)
            self.bounds = [compute_freq_bounds(data["S_db_full"], data["freqs_full"],
                                               data["sr"], s, e) for s, e in saved]
            note = f"loaded {len(saved)} saved segment(s)"
        else:
            self.segments = list(data["segments"])
            self.bounds = list(data["bounds"])
            note = f"detected {len(self.segments)} segment(s)"

        self.selected = 0 if self.segments else None
        self.dirty = False
        self._zoomed = False        # each file starts showing its full extent
        self._draw_audio()
        self._populate_freq_combo()
        self._redraw_segments()
        self._refresh_list()
        self.setWindowTitle(f"{APP_NAME} — {path}")
        self.statusBar().showMessage(f"{os.path.basename(path)} — {note}")
        # Cleared last: until segments/bounds are populated, a save would
        # write an empty .txt and silently mark the file as rejected.
        self._loading = False

    @QtCore.Slot(str, str, int)
    def _on_load_failed(self, path, msg, generation=None):
        if generation is not None and generation != self._load_generation:
            return
        self._loading = False
        self.audio = None
        self.segments, self.bounds, self.selected = [], [], None
        self._redraw_segments()
        self.info_label.setText(f"Could not load {os.path.basename(path)}: {msg}")
        self.statusBar().showMessage(f"Skipped {os.path.basename(path)}: {msg}")

    def _draw_audio(self):
        d = self.audio
        S, freqs, dur = d["S_db_full"], d["freqs_full"], d["duration"]
        self.spec_img.setImage(S, autoLevels=False, levels=(-60, 0))
        self.spec_img.setRect(QtCore.QRectF(0, float(freqs[0]), dur,
                                            float(freqs[-1] - freqs[0])))
        self.spec_img.setLookupTable(
            pg.colormap.get("plasma", source="matplotlib").getLookupTable(0.0, 1.0, 256))

        # Widen the pan limits to this file BEFORE setting the range: limits
        # left over from a shorter previous file would otherwise clamp the new
        # view and the file would not open at full extent.
        self.spec_plot.setLimits(xMin=0, xMax=dur)
        self.wave_plot.setLimits(xMin=0, xMax=dur)
        self.spec_plot.setXRange(0, dur, padding=0)

        # Min/max envelope: plain subsampling drops peaks and renders dense
        # audio as a solid block, so each output column keeps its true extremes.
        y = d["y"]
        target_cols = 3000
        if len(y) > target_cols * 2:
            per = len(y) // target_cols
            trimmed = y[:per * target_cols].reshape(target_cols, per)
            lo, hi = trimmed.min(axis=1), trimmed.max(axis=1)
            t = np.repeat(np.linspace(0, dur, target_cols), 2)
            amp = np.empty(target_cols * 2, dtype=float)
            amp[0::2], amp[1::2] = lo, hi
        else:
            t = np.linspace(0, dur, len(y))
            amp = y

        self.wave_curve.setData(t, amp)
        self.wave_plot.setXRange(0, dur, padding=0)

        # These recordings have crest factors around 20-25, so scaling to the
        # absolute peak flattens the actual vocalisation into a line near zero.
        # Scale to a high percentile instead and let rare transients clip.
        scale = float(np.percentile(np.abs(y), 99.5))
        if scale <= 0:
            scale = float(np.abs(y).max()) or 1.0
        self.wave_plot.setYRange(-scale * 1.3, scale * 1.3, padding=0)

    def _populate_freq_combo(self):
        sr = self.audio["sr"]
        steps = [v for v in MP3_NYQUIST_VALUES if FREQ_MIN < v <= sr / 2] or [int(sr / 2)]
        self.freq_combo.blockSignals(True)
        self.freq_combo.clear()
        self.freq_combo.addItems([str(v) for v in steps])
        default = min(steps, key=lambda v: abs(v - DEFAULT_FREQ_CUTOFF))
        self.freq_combo.setCurrentIndex(steps.index(default))
        self.freq_combo.blockSignals(False)
        self.spec_plot.setYRange(FREQ_MIN, default, padding=0)

    def _apply_freq_cutoff(self):
        if self.audio and self.freq_combo.currentText():
            self.spec_plot.setYRange(FREQ_MIN, int(self.freq_combo.currentText()),
                                     padding=0)

    # ── Segment rendering ────────────────────────────────────────────────
    def _redraw_segments(self):
        for it in self.region_items:
            self.spec_plot.removeItem(it)
        for it in self.label_items:
            self.spec_plot.removeItem(it)
        for it in self.wave_items:
            self.wave_plot.removeItem(it)
        self.region_items.clear()
        self.label_items.clear()
        self.wave_items.clear()

        if len(self.segments) > MAX_SEGMENTS:
            del self.segments[MAX_SEGMENTS:]
            del self.bounds[MAX_SEGMENTS:]
        if not self.segments:
            self.selected = None
        elif self.selected is None or not (0 <= self.selected < len(self.segments)):
            self.selected = 0

        for i, (start, _end) in enumerate(self.segments):
            sel = (i == self.selected)
            fmin, fmax = (self.bounds[i] if i < len(self.bounds)
                          else (FREQ_MIN, self._nyquist()))
            box = SegmentBox(i, start, fmin, fmax, selected=sel, owner=self)
            self.spec_plot.addItem(box)
            self.region_items.append(box)

            # anchor=(1,0) puts the badge's top-right at the box's top-right,
            # so the number sits inside the box. It stays in the corner rather
            # than centred to keep the call energy in the middle readable, and
            # the fill stays opaque so it reads over a bright spectrogram.
            label = pg.TextItem(str(i + 1), anchor=(1, 0),
                                color=LABEL_TEXT_COLOR,
                                fill=LABEL_FILL_SELECTED if sel else LABEL_FILL_NORMAL)
            label.setFont(_label_font())
            label.setPos(start + SEGMENT_DURATION, fmax)
            label.setZValue(11)
            self.spec_plot.addItem(label)
            self.label_items.append(label)

            span = WaveSpan(i, start, selected=sel, owner=self)
            self.wave_plot.addItem(span)
            self.wave_items.append(span)

        self._update_info()

    def _nyquist(self):
        return self.audio["sr"] / 2 if self.audio else DEFAULT_FREQ_CUTOFF

    def preview_segment_move(self, i, new_start):
        """Live feedback during a drag — moves the box only, no state change."""
        if not (0 <= i < len(self.segments)) or i >= len(self.region_items):
            return
        lo = self._clamp_start(i, new_start)
        self.region_items[i].set_start(lo)
        if i < len(self.label_items):
            fmax = self.bounds[i][1] if i < len(self.bounds) else self._nyquist()
            self.label_items[i].setPos(lo + SEGMENT_DURATION, fmax)
        if i < len(self.wave_items):
            self.wave_items[i].set_start(lo)

    def commit_segment_move(self, i, new_start):
        """Persist a finished drag.

        The redraw is deferred: this runs inside the box's own mouse handler,
        and rebuilding the items synchronously would delete the object whose
        event is still executing.
        """
        if not (0 <= i < len(self.segments)):
            return
        lo = self._clamp_start(i, new_start)
        if abs(lo - self.segments[i][0]) > 1e-6:
            self.stop_playback()
            self.segments[i] = (lo, lo + SEGMENT_DURATION)
            self.dirty = True
        self.selected = i
        # Highlight now, full rebuild after this event handler returns.
        self._apply_selection_highlight()
        self._update_info()
        QtCore.QTimer.singleShot(0, self._redraw_segments)

    def _clamp_start(self, idx, new_start):
        """Mirrors Stage 4's clamp: neighbours bound the move, but a segment
        flush against one falls back to the audio bounds rather than freezing."""
        dur = self.audio["duration"] if self.audio else SEGMENT_DURATION
        hard_lo, hard_hi = 0.0, max(0.0, dur - SEGMENT_DURATION)
        left = self.segments[idx - 1][1] if idx > 0 else hard_lo
        right = (self.segments[idx + 1][0] - SEGMENT_DURATION
                 if idx < len(self.segments) - 1 else hard_hi)
        if right <= left:
            left, right = hard_lo, hard_hi
        return max(left, min(new_start, right))

    def _update_info(self):
        if not self.audio:
            self.info_label.setText("No file loaded.")
            return
        snr = calculate_enhanced_snr(self.audio["y"], self.audio["sr"], self.segments)
        sel = (f"segment {self.selected + 1}/{len(self.segments)}"
               if self.selected is not None else "none")
        pos = f"file {self.index + 1}/{len(self.entries)}"
        flag = "  •unsaved" if self.dirty else ""
        self.info_label.setText(
            f"{pos} | segments: {len(self.segments)} | playing: {sel} | "
            f"avg SNR {snr['average_snr']:.1f} dB | peak {snr['peak_snr']:.1f} dB{flag}")

    # ── Selection / editing ──────────────────────────────────────────────
    # ── Zoom ─────────────────────────────────────────────────────────────
    def zoom_to_selected(self, context=2.0):
        """Frame the selected segment with `context` seconds either side.

        On a long recording a 3s segment is a few pixels wide, which is what
        previously forced these files out to an external editor.
        """
        if not self.audio or self.selected is None:
            return
        start, end = self.segments[self.selected]
        dur = self.audio["duration"]
        lo = max(0.0, start - context)
        hi = min(dur, end + context)
        self.spec_plot.setXRange(lo, hi, padding=0)
        self._zoomed = True
        self.statusBar().showMessage(
            f"Zoomed to segment {self.selected + 1} ({lo:.1f}–{hi:.1f}s of {dur:.0f}s)")

    def zoom_full(self):
        """Show the whole file again."""
        if not self.audio:
            return
        self.spec_plot.setXRange(0, self.audio["duration"], padding=0)
        self._zoomed = False
        self.statusBar().showMessage(f"Showing full file ({self.audio['duration']:.0f}s)")

    def zoom_by(self, factor):
        """Zoom in/out about the centre of the current view."""
        if not self.audio:
            return
        lo, hi = self.spec_plot.viewRange()[0]
        dur = self.audio["duration"]
        centre = (lo + hi) / 2.0
        half = max(0.5, min((hi - lo) * factor / 2.0, dur / 2.0))
        new_lo, new_hi = centre - half, centre + half
        if new_lo < 0:
            new_lo, new_hi = 0.0, min(dur, 2 * half)
        if new_hi > dur:
            new_lo, new_hi = max(0.0, dur - 2 * half), dur
        self.spec_plot.setXRange(new_lo, new_hi, padding=0)
        self._zoomed = (new_hi - new_lo) < dur - 1e-6
        self.statusBar().showMessage(
            f"View {new_lo:.1f}–{new_hi:.1f}s of {dur:.0f}s")

    def _follow_selection(self):
        """Keep the selected segment in view when zoomed in."""
        if not getattr(self, "_zoomed", False) or self.selected is None or not self.audio:
            return
        start, end = self.segments[self.selected]
        lo, hi = self.spec_plot.viewRange()[0]
        if start >= lo and end <= hi:
            return                      # already visible, don't jolt the view
        self.zoom_to_selected(context=max(2.0, (hi - lo - SEGMENT_DURATION) / 2.0))

    def select_segment(self, idx):
        """Select a segment and repaint the highlight immediately.

        The highlight must not lag behind self.selected: if it did, pressing
        Space or Delete right after a click would act on the new selection
        while the screen still showed the old one, which reads as the wrong
        segment being played or deleted. Only full rebuilds are deferred;
        recolouring is cheap and safe to do inline.
        """
        if not self.segments or not (0 <= idx < len(self.segments)):
            return
        self.selected = idx
        self._apply_selection_highlight()
        self._update_info()
        self._follow_selection()

    def _apply_selection_highlight(self):
        """Sync every box's selected flag to self.selected and repaint."""
        for i, box in enumerate(self.region_items):
            box.set_selected(i == self.selected)
        for i, span in enumerate(self.wave_items):
            span.set_selected(i == self.selected)
        for i, label in enumerate(self.label_items):
            # TextItem has no setFill(); assigning then re-setting the text is
            # what makes it repaint. setText() also resets the colour, so the
            # text colour has to be passed again here.
            label.fill = (LABEL_FILL_SELECTED if i == self.selected
                          else LABEL_FILL_NORMAL)
            label.setText(str(i + 1), color=LABEL_TEXT_COLOR)
            label.setFont(_label_font())
            label.update()

    def step_selection(self, delta):
        if not self.segments:
            return
        base = 0 if self.selected is None else self.selected
        self.select_segment((base + delta) % len(self.segments))

    def delete_selected(self):
        if self.selected is None or not self.segments:
            return
        self.stop_playback()
        i = self.selected
        del self.segments[i]
        if i < len(self.bounds):
            del self.bounds[i]
        if not self.segments:
            self.selected = None
        else:
            self.selected = min(i, len(self.segments) - 1)
        self.dirty = True
        self._redraw_segments()

    def rerun_detection(self):
        """Re-detect with the current slider settings.

        Moving a slider always replaces the segments outright — that is the
        point of the control, so it never asks for confirmation. Edits made
        before the adjustment are discarded by design.
        """
        if not self.audio:
            return
        self.stop_playback()
        self._retire_thread("_dthread")
        self._detect_generation = getattr(self, "_detect_generation", 0) + 1
        self._discarded_edits = self.dirty
        self.statusBar().showMessage("Re-running detection …")
        self._dthread = QtCore.QThread(self)
        self._dworker = DetectWorker(
            self.audio["S_det"], self.audio["S_db_full"], self.audio["freqs_full"],
            self.audio["sr"], self.audio["duration"],
            self.sl_thresh.value() / 100.0,
            self.sl_erode.value(), self.sl_dilate.value(),
            self._detect_generation)
        self._dworker.moveToThread(self._dthread)
        self._dthread.started.connect(self._dworker.run)
        self._dworker.finished.connect(self._on_detected,
                                       QtCore.Qt.QueuedConnection)
        self._dworker.finished.connect(self._dthread.quit)
        self._dthread.start()

    @QtCore.Slot(object)
    @QtCore.Slot(object, int)
    def _on_detected(self, result, generation=None):
        # A superseded detection (slider moved again, or file changed) must
        # not overwrite the current segments.
        if generation is not None and generation != getattr(self, "_detect_generation", 0):
            return
        segs, bounds = result
        self.segments, self.bounds = list(segs), list(bounds)
        self.selected = 0 if self.segments else None
        self.dirty = False
        self._redraw_segments()
        note = " (previous edits discarded)" if getattr(self, "_discarded_edits", False) else ""
        self._discarded_edits = False
        self.statusBar().showMessage(f"Detected {len(segs)} segment(s){note}")

    # ── Playback ─────────────────────────────────────────────────────────
    def toggle_play(self):
        if self.play_timer.isActive():
            self.stop_playback()
            return
        if not self.audio or self.selected is None:
            return
        start, end = self.segments[self.selected]
        sr = self.audio["sr"]
        a, b = max(0, int(start * sr)), min(len(self.audio["y"]), int(end * sr))
        if b <= a:
            return
        try:
            sd.play(self.audio["y"][a:b], sr)
        except Exception as exc:                      # noqa: BLE001
            # No/!misconfigured output device — common on headless Linux.
            # Annotation is still usable without audio, so report and continue.
            self.statusBar().showMessage(f"Audio playback unavailable: {exc}")
            return
        self.play_span = (start, end)
        self.play_clock = QtCore.QElapsedTimer()
        self.play_clock.start()
        self.spec_cursor.setVisible(True)
        self.wave_cursor.setVisible(True)
        self.play_button.setText("■ Stop (Space)")
        self.play_timer.start()

    def stop_playback(self):
        if self.play_timer.isActive():
            self.play_timer.stop()
        try:
            sd.stop()
        except Exception:                             # noqa: BLE001
            pass
        self.spec_cursor.setVisible(False)
        self.wave_cursor.setVisible(False)
        if hasattr(self, "play_button"):
            self.play_button.setText("▶ Play (Space)")

    def _tick_playback(self):
        if not self.play_span:
            return
        start, end = self.play_span
        t = start + self.play_clock.elapsed() / 1000.0
        if t >= end:
            self.stop_playback()
            return
        self.spec_cursor.setPos(t)
        self.wave_cursor.setPos(t)

    # ── Save / reject ────────────────────────────────────────────────────
    def _write_annotation(self, segments):
        path = self.entries[self.index].path
        txt = annotation_path_for(path)
        xc_type = lookup_xc_type(path)
        with open(txt, "w") as f:
            for start, end in segments[:MAX_SEGMENTS]:
                f.write(f"{start:.6f}\t{end:.6f}\t{xc_type}\n")
        self.entries[self.index].refresh()
        self.dirty = False
        return txt

    def save_and_next(self):
        if self._loading or not self.audio or self.index < 0:
            self.statusBar().showMessage("Still loading — not saved.")
            return
        self.stop_playback()
        txt = self._write_annotation(self.segments)
        self.statusBar().showMessage(
            f"Saved {len(self.segments)} segment(s) → {os.path.basename(txt)}")
        self._refresh_list()
        self._advance()

    def reject_and_next(self):
        if self._loading or self.index < 0:
            self.statusBar().showMessage("Still loading — not rejected.")
            return
        self.stop_playback()
        txt = self._write_annotation([])      # zero-length == rejected
        self.statusBar().showMessage(f"Rejected → {os.path.basename(txt)}")
        self._refresh_list()
        self._advance()

    def _advance(self):
        """Move to the next file after a save/reject.

        Reaching the end is a normal outcome, not a reason to close: the
        window stays open on the last file so the result can still be seen
        and revised.
        """
        nxt = next_unevaluated(self.entries, self.index)
        if nxt is None:
            nxt = self.index + 1
        if nxt >= len(self.entries):
            remaining = sum(1 for e in self.entries
                            if e.status is Status.TO_EVALUATE)
            if remaining:
                # Wrap around to catch anything skipped earlier in the pass.
                first = next_unevaluated(self.entries, -1)
                if first is not None:
                    self.load_index(first)
                    return
            self.statusBar().showMessage(
                f"End of queue — {format_summary(self.entries)}. "
                f"Nothing left to evaluate; window stays open.")
            return
        self.load_index(nxt)

    def next_file(self):
        if self.index + 1 < len(self.entries):
            self.load_index(self.index + 1)

    def prev_file(self):
        if self.index > 0:
            self.load_index(self.index - 1)

    # ── Misc ─────────────────────────────────────────────────────────────
    def _confirm_discard(self, msg="This file has unsaved changes."):
        reply = QtWidgets.QMessageBox.question(
            self, "Discard changes?", msg + "\n\nDiscard them?",
            QtWidgets.QMessageBox.Discard | QtWidgets.QMessageBox.Cancel)
        return reply == QtWidgets.QMessageBox.Discard

    @staticmethod
    def _load_state():
        try:
            with open(STATE_FILE) as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    @classmethod
    def _write_state(cls, **updates):
        """Merge keys into the state file, preserving everything else."""
        state = cls._load_state()
        state.update(updates)
        try:
            with open(STATE_FILE, "w") as f:
                json.dump(state, f, indent=2)
        except OSError:
            pass

    @classmethod
    def _last_dir(cls):
        return cls._load_state().get("last_dir", "")

    @classmethod
    def _save_last_dir(cls, path):
        cls._write_state(last_dir=path)

    def _centre_on_primary(self):
        """Centre the default-sized window on the primary display.

        Used when there is no usable saved geometry; Qt would otherwise leave
        the window at (0,0), which can sit under the menu bar or panel.
        """
        try:
            screen = (QtGui.QGuiApplication.primaryScreen()
                      or next(iter(QtGui.QGuiApplication.screens()), None))
            if screen is None:
                return
            area = screen.availableGeometry()
            if area.width() <= 0 or area.height() <= 0:
                return
            w = min(self.width(), area.width())
            h = min(self.height(), area.height())
            self.setGeometry(area.x() + (area.width() - w) // 2,
                             area.y() + (area.height() - h) // 2, w, h)
        except Exception:                             # noqa: BLE001
            pass

    def _save_geometry(self):
        """Persist size, position and which screen the window is on.

        Never allowed to fail: a window-state write must not prevent the app
        from closing, whatever the platform or display setup does.
        """
        try:
            g = self.normalGeometry()
            if self.isMaximized() or not g.isValid() or g.width() <= 0:
                g = self.geometry()
            screen = self.screen()
            self._write_state(window={
                "x": int(g.x()), "y": int(g.y()),
                "width": int(g.width()), "height": int(g.height()),
                "maximized": bool(self.isMaximized()),
                "screen": screen.name() if screen is not None else None,
                # Screen names are platform-specific ('LG IPS QHD' on macOS,
                # 'DP-1' on X11, '\\\\.\\DISPLAY1' on Windows), so record the
                # platform and only trust the name when it matches.
                "platform": sys.platform,
            })
        except Exception as exc:                      # noqa: BLE001
            print(f"  (could not save window geometry: {exc})")

    def _restore_geometry(self):
        """Restore the previous size/screen, ignoring anything unusable now.

        Falls back safely at every step: a missing screen, a different OS, a
        compositor that ignores window placement (common on Wayland), or any
        Qt error leaves the default size in place rather than crashing or
        opening the window somewhere invisible.
        """
        try:
            win = self._load_state().get("window")
            if not isinstance(win, dict):
                return False
            try:
                w = int(win.get("width", 0))
                h = int(win.get("height", 0))
                x = int(win.get("x", 0))
                y = int(win.get("y", 0))
            except (TypeError, ValueError):
                return False
            if w < 600 or h < 400:
                return False

            screens = QtGui.QGuiApplication.screens() or []
            primary = QtGui.QGuiApplication.primaryScreen()
            if not screens and primary is None:
                return False            # no usable display information

            # Only match by name when the state came from this same platform.
            target = None
            if win.get("platform") == sys.platform:
                saved_name = win.get("screen")
                if saved_name:
                    target = next((s for s in screens
                                   if s.name() == saved_name), None)

            centre_only = target is None
            if target is None:
                target = primary or screens[0]
            area = target.availableGeometry()
            if area.width() <= 0 or area.height() <= 0:
                return False

            saved_w, saved_h = w, h
            w, h = min(w, area.width()), min(h, area.height())
            if (w, h) != (saved_w, saved_h):
                # The saved size cannot fit the display in use now, so it is
                # reduced rather than restored. Say so: otherwise the window
                # just appears the wrong size with no explanation.
                print(f"  (saved window {saved_w}x{saved_h} does not fit "
                      f"{area.width()}x{area.height()} on "
                      f"'{target.name() or 'display'}', using {w}x{h})")
            if centre_only:
                # Unknown/!absent screen, or state from another OS: keep the
                # size but centre it, since the saved coordinates mean nothing.
                x = area.x() + (area.width() - w) // 2
                y = area.y() + (area.height() - h) // 2
            else:
                # Clamp so the title bar always stays reachable. Negative
                # coordinates are legitimate for monitors left of / above the
                # primary one on Windows and X11, so clamp to the screen's own
                # bounds rather than to zero.
                x = max(area.x(), min(x, area.x() + area.width() - w))
                y = max(area.y(), min(y, area.y() + area.height() - h))

            self.setGeometry(int(x), int(y), int(w), int(h))
            if win.get("maximized"):
                self.showMaximized()
            return True
        except Exception as exc:                      # noqa: BLE001
            print(f"  (could not restore window geometry, using default: {exc})")
            return False

    def _save_session(self):
        """Persist folder + window state. Safe to call more than once."""
        try:
            # Prefer the folder that was opened. Falling back to the first
            # entry's directory would save a subfolder of it, because the
            # scan is recursive, and the next launch would then reopen that
            # subfolder instead of the chosen root.
            folder = getattr(self, "root_folder", None)
            if not folder and self.entries:
                folder = os.path.dirname(self.entries[0].path)
            if folder and os.path.isdir(folder):
                self._save_last_dir(folder)
        except Exception:                             # noqa: BLE001
            pass
        self._save_geometry()

    def closeEvent(self, event):
        self.stop_playback()
        self._save_session()
        self._wait_for_thread("_thread")
        self._wait_for_thread("_dthread")
        super().closeEvent(event)


def parse_args():
    p = argparse.ArgumentParser(description="Qt annotator for Stage 4 segments")
    if get_profile is not None:
        p.add_argument("--dataset", choices=["mygardenbird", "sea-bird30"],
                       default=os.environ.get("PIPELINE_DATASET", "mygardenbird"))
    p.add_argument("--folder", default=None, help="Folder of audio files to queue")
    p.add_argument("--file", default=None, help="Open a single file directly")
    return p.parse_args()


CRASH_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "stage4b_errors.log")


def _install_excepthook():
    """Log uncaught exceptions instead of letting them kill the session.

    An exception raised inside a Qt event handler unwinds into C++, where
    Python's default hook may print to a terminal that is not being watched
    and the process can abort outright. Annotation state is in memory and a
    half-finished file is worth more than a clean stack trace, so the error is
    recorded and the event loop continues.
    """
    previous = sys.excepthook

    def hook(exc_type, exc, tb):
        if issubclass(exc_type, KeyboardInterrupt):
            previous(exc_type, exc, tb)
            return
        text = "".join(traceback.format_exception(exc_type, exc, tb))
        stamp = datetime.now().isoformat(timespec="seconds")
        try:
            with open(CRASH_LOG, "a") as f:
                f.write(f"\n===== {stamp} =====\n{text}")
        except OSError:
            pass
        sys.stderr.write(f"\n!! error logged to {CRASH_LOG}\n{text}")

    sys.excepthook = hook


def _install_sigint(app):
    """Make Ctrl+C in the launching terminal quit the app cleanly.

    Two things are in the way. Python only runs signal handlers between
    bytecodes, and while Qt sits idle in its C++ event loop no bytecode runs,
    so a handler registered here would not fire until the next mouse or key
    event. A repeating timer that does nothing is enough to break that: each
    tick re-enters the interpreter and any pending signal is delivered.

    The handler then asks Qt to quit rather than raising, so shutdown goes
    through the normal path and aboutToQuit still saves the session.
    """
    def handler(signum, frame):                       # noqa: ARG001
        print("\n  interrupt received, saving session and quitting...")
        app.quit()

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)

    # Kept on the app so it is not garbage-collected; 200 ms is frequent
    # enough to feel immediate and far too cheap to matter.
    app._sigint_timer = QtCore.QTimer()
    app._sigint_timer.timeout.connect(lambda: None)
    app._sigint_timer.start(200)


def main():
    args = parse_args()
    _install_excepthook()
    app = QtWidgets.QApplication(sys.argv)
    _install_sigint(app)
    win = Annotator(folder=args.folder, single_file=args.file)
    win.show()
    print("  (Ctrl+C in this terminal quits and saves)")
    # Cmd+Q on macOS quits without delivering closeEvent to the window, so the
    # session state has to be saved from the application's own shutdown signal
    # as well.
    app.aboutToQuit.connect(win._save_session)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
