"""DOORS Sound Radar - passive audio overlay for Roblox DOORS.

Listens to your system output (WASAPI loopback - does NOT touch the game),
matches sounds against short reference samples you record once, and shows
the entity name + how to survive it in a floating window.

Record one clean sample per entity while you play (REC button, ~2-4s).
Rush and Ambush share a sound: repeated hits auto-report as Ambush.
"""

import os
import queue
import sys
import threading
import time
import wave

import numpy as np
import tkinter as tk
from tkinter import ttk

if getattr(sys, "frozen", False):
    APP_DIR = os.path.dirname(sys.executable)
else:
    APP_DIR = os.path.dirname(os.path.abspath(__file__))
TPL_DIR = os.path.join(APP_DIR, "templates")

KEEP_SECS = 25.0
HOP_SECS = 0.1
FIRE_MARGIN_SECS = 0.55
REC_SECS = 4.0
REC_MIN_SECS = 0.4
REC_MAX_SECS = 6.0
RMS_GATE = 0.0015

ENTITIES = {
    "Rush": dict(
        color="#ff4d4d", cooldown=3.5,
        tip="Hide in a locker/closet and close it. Stay in until the lights "
            "come back AND the sound has fully passed."),
    "Screech": dict(
        color="#ffd24a", cooldown=5.0,
        tip="In a dark room, when you hear the screech: TURN AROUND and CLICK "
            "it before it hits. Safer: stay in lamplight while looting."),
    "Eyes": dict(
        color="#a86bff", cooldown=4.0,
        tip="Do NOT look at it and do NOT attack. Look down/away and walk "
            "slowly past it."),
    "Seek": dict(
        color="#3bd6ff", cooldown=6.0,
        tip="Water/splash chase: RUN the path, jump gaps, slide under bars, "
            "dodge the lunges, hide in a locker when told."),
    "Figure": dict(
        color="#ff8c3b", cooldown=6.0,
        tip="Waltz/heartbeat: CROUCH while it tilts its head (it listens). "
            "When it screeches, hide under a table. Never run on debris."),
    "Electrical": dict(
        color="#4aff88", cooldown=2.5,
        tip="Zaps = Figment's line. Break line-of-sight down the hall, hide "
            "in a locker, dodge the spark tripwires."),
    "Timoth": dict(
        color="#8cff4a", cooldown=4.0,
        tip="Rustling near a flower: step AROUND the patch (or crouch). "
            "Walking into flowers triggers him."),
    "Halt": dict(
        color="#ff4ad0", cooldown=5.0,
        tip="Warped/distorted ringing: trust nothing you see. Walk BACK to "
            "the previous room and keep backtracking until it fades."),
    "Glitch": dict(
        color="#9ff0ff", cooldown=4.0,
        tip="Shimmering chime: harmless bonus room. Collect the gold and "
            "items and move on."),
}

AMBUSHER = dict(
    name="Ambush",
    color="#ff9d3b",
    tip="Same sound as Rush, but it passes 2-4 times. DO NOT peek early - "
        "stay hidden until the very last pass and the lights return.")

AMBUSH_WINDOW = 14.0


def save_wav(path, x, sr):
    x = np.clip(x, -1.0, 1.0)
    data = (x * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(data.tobytes())


def load_wav(path):
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        n = w.getnframes()
        raw = w.readframes(n)
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32767.0
    return x, sr


def sliding_ncc(x, t):
    """Normalized cross-correlation of template t at each offset of x."""
    L = len(t)
    Nx = len(x)
    if Nx < L:
        return np.zeros(0)
    N = 1 << (Nx + L - 2).bit_length()
    corr = np.fft.irfft(np.fft.rfft(x, N) * np.conj(np.fft.rfft(t, N)), N)[: Nx - L + 1]
    cs = np.concatenate(([0.0], np.cumsum(x * x)))
    energy = cs[L: L + corr.size] - cs[: corr.size]
    denom = np.linalg.norm(t) * np.sqrt(np.maximum(energy, 1e-12))
    return corr / denom


class Radar(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.events = queue.Queue()
        self.status = {"rms": 0.0, "detail": "starting...", "rec": None}
        self.threshold = 0.72
        self.templates = {}
        self._pending = None
        self._cancel = False
        self._cooldown_until = {}
        self._last_rush = 0.0
        self._buf = np.zeros(0)
        self._stop = threading.Event()
        self._blocks = queue.Queue(maxsize=64)
        self.sr = 48000
        self._pa = None
        self._sd_stream = None
        self._in_ch = 2

    def load_templates(self):
        os.makedirs(TPL_DIR, exist_ok=True)
        for name in ENTITIES:
            p = os.path.join(TPL_DIR, f"{name}.wav")
            if os.path.exists(p):
                try:
                    x, sr = load_wav(p)
                    if sr != self.sr:
                        x = np.interp(
                            np.arange(int(len(x) * self.sr / sr)),
                            np.arange(len(x)), x)
                    x = x - x.mean()
                    if len(x) >= REC_MIN_SECS * self.sr:
                        self.templates[name] = x
                except Exception as e:
                    self.events.put(("log", f"failed to load {name}: {e}"))

    def start_recording(self, name):
        if self._pending is not None:
            return
        frames = []
        back = int(2.5 * self.sr)
        if len(self._buf) >= back:
            frames.append(self._buf[-back:].copy())
        self._pending = [name, frames, REC_SECS * self.sr]
        self.status["rec"] = (name, REC_SECS)

    def stop_recording(self):
        if self._pending is not None:
            total = sum(len(f) for f in self._pending[1])
            if total >= REC_MIN_SECS * self.sr:
                self._finish_recording(keep=True)
            else:
                self._pending = None
                self.status["rec"] = None
                self.events.put(("log", "recording too short"))

    def _finish_recording(self, keep=True):
        name, frames, _ = self._pending
        self._pending = None
        self.status["rec"] = None
        x = np.concatenate(frames) if frames else np.zeros(0)
        x = x - x.mean()
        secs = len(x) / self.sr
        if not keep:
            return
        if len(x) < REC_MIN_SECS * self.sr:
            self.events.put(("log", "sample too short, try again"))
            return
        if secs > REC_MAX_SECS:
            x = x[: int(REC_MAX_SECS * self.sr)]
            secs = REC_MAX_SECS
        if np.sqrt(np.mean(x * x)) < 2e-4:
            self.events.put(("log", f"{name}: sample was (near) silent, try again"))
            return
        save_wav(os.path.join(TPL_DIR, f"{name}.wav"), x, self.sr)
        self.templates[name] = x
        self.events.put(("log", f"{name} sample saved ({secs:.1f}s)"))

    def stop(self):
        self._stop.set()

    def _open_stream(self):
        # Preferred: WASAPI loopback via PyAudioWPatch (captures output device).
        try:
            import pyaudiowpatch as pa
        except Exception:
            pa = None
        if pa is not None:
            try:
                self._pa = pa.PyAudio()
                info = self._pa.get_default_wasapi_loopback()
                self.sr = int(info["defaultSampleRate"])
                self._in_ch = int(info["maxInputChannels"])
                self._dev_name = info["name"]
                self._pa_stream = self._pa.open(
                    format=pa.paInt16, channels=self._in_ch, rate=self.sr,
                    input=True, frames_per_buffer=1024,
                    input_device_index=info["index"],
                    stream_callback=self._pa_cb)
                return
            except Exception:
                try:
                    self._pa.terminate()
                except Exception:
                    pass
                self._pa = None
        # Fallback: sounddevice (Stereo Mix / What U Hear style record device).
        import sounddevice as sd
        mix = None
        for i, d in enumerate(sd.query_devices()):
            if d["max_input_channels"] > 0 and (
                    "mix" in d["name"].lower()
                    or "what u hear" in d["name"].lower()):
                mix = (i, d)
                break
        if mix is None:
            raise RuntimeError(
                "No WASAPI loopback (install 'pyaudiowpatch') and no Stereo "
                "Mix record device found.")
        i, d = mix
        self.sr = int(d["default_samplerate"])
        self._in_ch = d["max_input_channels"]
        self._dev_name = d["name"]
        self._sd_stream = sd.InputStream(
            device=i, samplerate=self.sr, channels=self._in_ch,
            dtype="float32", blocksize=1024,
            callback=lambda indata, n, t, f: self._put(
                np.asarray(indata, np.float64).mean(axis=1)))
        self._sd_stream.start()

    def _pa_cb(self, in_data, frame_count, time_info, status_flags):
        x = np.frombuffer(in_data, dtype=np.int16).astype(np.float64) / 32768.0
        x = x.reshape(-1, self._in_ch).mean(axis=1)
        self._put(x)
        return (None, 0)

    def _put(self, block):
        if self._blocks.full():
            try:
                self._blocks.get_nowait()
            except queue.Empty:
                pass
        self._blocks.put(block)

    def run(self):
        try:
            self._open_stream()
        except Exception as e:
            self.events.put(("fatal", str(e)))
            return
        self.status["detail"] = f"listening: {self._dev_name} @ {self.sr}Hz"
        self.events.put(("log", self.status["detail"]))
        self.load_templates()
        last_detect = 0.0
        try:
            while not self._stop.is_set():
                try:
                    block = self._blocks.get(timeout=0.25)
                except queue.Empty:
                    if time.monotonic() - last_detect > 1.0:
                        self.status["rms"] = 0.0
                    continue
                self._pump(block)
                now = time.monotonic()
                if now - last_detect >= HOP_SECS:
                    last_detect = now
                    self._detect(now)
        finally:
            self._close_stream()

    def _close_stream(self):
        s = getattr(self, "_sd_stream", None)
        if s is not None:
            try:
                s.stop(); s.close()
            except Exception:
                pass
        p = getattr(self, "_pa", None)
        if p is not None:
            try:
                self._pa_stream.stop_stream()
                self._pa_stream.close()
                p.terminate()
            except Exception:
                pass

    def _pump(self, mono):
        if self._pending is not None:
            name, frames, target = self._pending
            frames.append(mono)
            if sum(len(f) for f in frames) >= target:
                self._finish_recording(keep=True)
        self._buf = np.concatenate((self._buf, mono))[-int(KEEP_SECS * self.sr):]
        self.status["rms"] = float(np.sqrt(np.mean(mono * mono)))

    def _detect(self, now):
        if not self.templates or self._pending is not None:
            return
        maxL = max(len(t) for t in self.templates.values())
        need = maxL + int((FIRE_MARGIN_SECS + 0.3) * self.sr)
        if len(self._buf) < need:
            return
        tail = self._buf[-need:]
        margin = int(FIRE_MARGIN_SECS * self.sr)
        for name, tpl in self.templates.items():
            if now < self._cooldown_until.get(name, 0.0):
                continue
            s = sliding_ncc(tail, tpl)
            if s.size == 0:
                continue
            i = int(np.argmax(s))
            score = float(s[i])
            if score < self.threshold:
                continue
            if len(tail) - (i + len(tpl)) > margin:
                continue
            self._fire(name, score, now)

    def _fire(self, name, score, now):
        self._cooldown_until[name] = now + ENTITIES[name]["cooldown"]
        shown, color, tip = name, ENTITIES[name]["color"], ENTITIES[name]["tip"]
        if name == "Rush":
            if now - self._last_rush < AMBUSH_WINDOW:
                shown, color, tip = (AMBUSHER["name"], AMBUSHER["color"],
                                     AMBUSHER["tip"])
            self._last_rush = now
        self.events.put(("alert", shown, color, tip, score, now))

    @property
    def template_names(self):
        return sorted(self.templates)


class Overlay:
    BG = "#101218"
    FG = "#e8eaf2"

    def __init__(self, radar):
        self.radar = radar
        self.root = tk.Tk()
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        self.root.attributes("-alpha", 0.94)
        self.root.configure(bg=self.BG)
        self.W = 344
        sw = self.root.winfo_screenwidth()
        self.root.geometry(f"+{sw - self.W - 16}+52")
        self._drag = None
        self._alert_since = 0.0
        self._collapsed = False
        self._rec_btns = {}
        self._tpl_lbls = {}
        self._build()
        self.root.protocol("WM_DELETE_WINDOW", self.quit)
        self.root.bind("<Escape>", lambda e: self.quit())
        self.root.bind("<Return>", lambda e: self._send_alert())
        self.root.after(80, self._poll)

    def _build(self):
        r = self.root
        head = tk.Frame(r, bg="#1a1e2a")
        head.pack(fill="x")
        head.bind("<ButtonPress-1>", self._drag_start)
        head.bind("<B1-Motion>", self._drag_move)
        tk.Label(head, text="DOORS Sound Radar", bg="#1a1e2a", fg=self.FG,
                 font=("Segoe UI", 11, "bold")).pack(side="left", padx=8, pady=4)
        tk.Button(head, text="x", command=self.quit, bg="#1a1e2a", fg="#8890a4",
                  relief="flat", font=("Segoe UI", 10), width=2).pack(side="right")
        tk.Button(head, text="_", command=self._collapse, bg="#1a1e2a",
                  fg="#8890a4", relief="flat", font=("Segoe UI", 10),
                  width=2).pack(side="right")

        self.banner = tk.Frame(r, bg=self.BG, height=118)
        self.banner.pack_propagate(False)
        self.banner.pack(fill="x")
        self.al_name = tk.Label(self.banner, text="standby - waiting for sounds",
                                bg=self.BG, fg="#5a6178",
                                font=("Segoe UI", 17, "bold"))
        self.al_name.pack(anchor="w", padx=10, pady=(8, 0))
        self.al_tip = tk.Label(self.banner, text="", bg=self.BG, fg=self.FG,
                               font=("Segoe UI", 9), wraplength=self.W - 24,
                               justify="left")
        self.al_tip.pack(anchor="w", padx=10)
        self.al_info = tk.Label(self.banner, text="", bg=self.BG, fg="#5a6178",
                                font=("Segoe UI", 8))
        self.al_info.pack(anchor="w", padx=10)

        meter = tk.Canvas(r, height=8, bg="#0a0c10", highlightthickness=0)
        meter.pack(fill="x")
        self.meter = meter
        self._meter_id = meter.create_rectangle(0, 4, 0, 4, fill="#2f8f4e",
                                                width=9, outline="#2f8f4e")

        opt = tk.Frame(r, bg=self.BG)
        opt.pack(fill="x", padx=8)
        tk.Label(opt, text="sensitivity", bg=self.BG, fg="#8890a4",
                 font=("Segoe UI", 8)).pack(side="left")
        self.sens = tk.Scale(opt, from_=0, to=100, orient="horizontal",
                             bg=self.BG, fg=self.FG, highlightthickness=0,
                             troughcolor="#252a38", length=120, showvalue=0,
                             command=self._sens_changed)
        self.sens.set(40)
        self.sens.pack(side="left", padx=6)
        tk.Button(opt, text="TEST", command=self._send_alert, bg="#252a38",
                  fg=self.FG, relief="flat", font=("Segoe UI", 8)).pack(
            side="right")

        body = tk.Frame(r, bg=self.BG)
        body.pack(fill="both", expand=True, padx=8, pady=4)
        tk.Label(body, text="hear a sound? click REC even up to 2s LATE - it "
                            "grabs the last 2.5s + next 1.5s and learns it. "
                            "One sample per entity, saved forever.",
                 bg=self.BG, fg="#6a7290", font=("Segoe UI", 8),
                 wraplength=self.W - 26, justify="left").pack(anchor="w")
        for name in ENTITIES:
            row = tk.Frame(body, bg=self.BG)
            row.pack(fill="x", pady=1)
            tk.Label(row, text=name, fg=ENTITIES[name]["color"], bg=self.BG,
                     font=("Segoe UI", 9, "bold"), width=10,
                     anchor="w").pack(side="left")
            lbl = tk.Label(row, text="no sample", fg="#5a6178", bg=self.BG,
                           font=("Segoe UI", 8), width=13, anchor="w")
            lbl.pack(side="left")
            self._tpl_lbls[name] = lbl
            btn = tk.Button(row, text="REC", bg="#252a38", fg=self.FG,
                            relief="flat", font=("Segoe UI", 8), width=7,
                            command=lambda n=name: self._rec(n))
            btn.pack(side="right")
            self._rec_btns[name] = btn

        self.log = tk.Label(r, text="", bg=self.BG, fg="#6a7290", anchor="w",
                            font=("Segoe UI", 8))
        self.log.pack(fill="x", padx=10, pady=(0, 6))
        self._packables = [
            (self.banner, dict(fill="x")),
            (meter, dict(fill="x")),
            (opt, dict(fill="x")),
            (body, dict(fill="both", expand=True)),
            (self.log, dict(fill="x")),
        ]

    def _collapse(self):
        self._collapsed = not self._collapsed
        if self._collapsed:
            for w, _ in self._packables:
                w.pack_forget()
        else:
            for w, kw in self._packables:
                w.pack(**kw)

    def _sens_changed(self, val):
        self.radar.threshold = 0.92 - int(val) / 100.0 * 0.38

    def _rec(self, name):
        if self.radar.status["rec"] is not None:
            self.radar.stop_recording()
        else:
            self.radar.start_recording(name)

    def _drag_start(self, e):
        self._drag = (e.x_root - self.root.winfo_x(),
                      e.y_root - self.root.winfo_y())

    def _drag_move(self, e):
        if self._drag:
            self.root.geometry(
                f"+{e.x_root - self._drag[0]}+{e.y_root - self._drag[1]}")

    def _send_alert(self):
        import random
        n = random.choice(list(ENTITIES) + ["Ambush"])
        if n == "Ambush":
            ev = (AMBUSHER["name"], AMBUSHER["color"], AMBUSHER["tip"])
        else:
            ev = (n, ENTITIES[n]["color"], ENTITIES[n]["tip"])
        self._show_alert(ev[0], ev[1], ev[2], 1.0)

    def _show_alert(self, name, color, tip, score):
        self.al_name.config(text=name.upper(), fg=color)
        self.al_tip.config(text=tip)
        self._alert_since = time.time()
        self._alert_score = score

    def _poll(self):
        while True:
            try:
                ev = self.radar.events.get_nowait()
            except queue.Empty:
                break
            if ev[0] == "alert":
                _, name, color, tip, score, _ = ev
                self._show_alert(name, color, tip, score)
            elif ev[0] == "log":
                self.log.config(text=ev[1])
                print("[radar]", ev[1])
            elif ev[0] == "fatal":
                self.al_name.config(text="AUDIO ERROR", fg="#ff4d4d")
                self.al_tip.config(text=ev[1])
                print("[fatal]", ev[1])

        st = self.radar.status
        rms = min(1.0, (st["rms"] or 0.0) ** 0.5 * 3.2)
        color = ("#2f8f4e" if rms < 0.6 else "#c8a13a" if rms < 0.9
                 else "#ff4d4d")
        self.meter.itemconfigure(self._meter_id, width=9, outline=color,
                                 fill=color)
        self.meter.coords(self._meter_id, 0, 4, max(0, rms * self.W), 4)

        rec = st["rec"]
        for name, btn in self._rec_btns.items():
            if rec and rec[0] == name:
                done = sum(len(f) for f in
                           self.radar._pending[1]) / self.radar.sr if \
                    self.radar._pending else 0
                btn.config(text=f"STOP {done:.1f}s", bg="#5a2430")
            else:
                btn.config(text="REC", bg="#252a38")

        for name, lbl in self._tpl_lbls.items():
            if name in self.radar.templates:
                secs = len(self.radar.templates[name]) / self.radar.sr
                lbl.config(text=f"sample {secs:.1f}s", fg="#4aff88")
            else:
                lbl.config(text="no sample", fg="#5a6178")

        if self._alert_since:
            age = time.time() - self._alert_since
            self.al_info.config(
                text=f"confidence {self._alert_score:.0%} - {age:.0f}s ago")
        else:
            self.al_info.config(text=st["detail"])
        self.root.after(80, self._poll)

    def quit(self):
        self.root.destroy()

    def run(self):
        self.root.mainloop()
        self.radar.stop()


def main():
    if not getattr(sys, "frozen", False):
        missing = [m for m in ("numpy", "pyaudiowpatch") if _find_spec(m) is None]
        if missing:
            print("missing deps. run:  pip install " + " ".join(missing))
            return 1
    os.makedirs(TPL_DIR, exist_ok=True)
    radar = Radar()
    radar.start()
    Overlay(radar).run()
    return 0


def _find_spec(name):
    import importlib.util
    try:
        return importlib.util.find_spec(name)
    except Exception:
        return None


if __name__ == "__main__":
    raise SystemExit(main())
