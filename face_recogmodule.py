# face_recogmodule.py
import os
import cv2
import numpy as np
from PIL import Image, ImageTk, Image as PILImage
import face_recognition

PEOPLE_DIR = "People"
TOLERANCE = 0.45

# ---- optional backends ----
_BACKENDS = []  # filled below if importable
try:
    import mss  # BGRA frames
    _BACKENDS.append("mss")
except Exception:
    pass
try:
    import dxcam  # RGB frames via Desktop Duplication (Windows)
    _BACKENDS.append("dxcam")
except Exception:
    pass
try:
    from PIL import ImageGrab  # RGB frames
    _BACKENDS.append("imagegrab")
except Exception:
    pass
try:
    import pyautogui  # RGB frames
    _BACKENDS.append("pyautogui")
except Exception:
    pass


def load_known_faces(people_dir=PEOPLE_DIR):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    encs, names = [], []
    if not os.path.isdir(people_dir):
        print(f"[faces] dir not found: {people_dir}")
        return encs, names
    for fname in sorted(os.listdir(people_dir)):
        base, ext = os.path.splitext(fname)
        if ext.lower() not in exts:
            continue
        try:
            img_rgb = np.ascontiguousarray(np.array(PILImage.open(os.path.join(people_dir, fname)).convert("RGB"), dtype=np.uint8))
            fe = face_recognition.face_encodings(img_rgb)
            if fe:
                name = "".join(c for c in base if not c.isdigit()).strip() or base
                encs.append(fe[0]); names.append(name)
        except Exception as e:
            print(f"[faces] skip {fname}: {e}")
    print(f"[faces] loaded: {len(names)}")
    return encs, names


def confidence(d, tol=TOLERANCE):
    return float(np.clip(1 - (d / max(tol, 1e-6)), 0.0, 1.0))


class _Grabber:
    """Unified screen grabber with multiple backends."""
    def __init__(self, backend_order=None):
        self.backend = None
        self.ctx = None
        self.region = None  # (left, top, width, height)
        self.backend_order = backend_order or _BACKENDS[:] or []
        # prefer dxcam on Windows
        if "dxcam" in self.backend_order:
            self.backend_order = ["dxcam"] + [b for b in self.backend_order if b != "dxcam"]

    def start(self):
        for b in self.backend_order:
            try:
                if b == "mss":
                    self.ctx = mss.mss()
                    info = self.ctx.monitors[0]  # virtual screen (reliable)
                    self.region = (info["left"], info["top"], info["width"], info["height"])
                    self.backend = "mss"
                    print(f"[grab] backend=mss region={self.region}")
                    return True
                elif b == "dxcam":
                    cam = dxcam.create(output_idx=0)  # primary monitor
                    ok = cam.start(target_fps=30, video_mode=True)
                    if not ok:
                        raise RuntimeError("dxcam start failed")
                    # infer size from first frame
                    frame = cam.get_latest_frame()
                    if frame is None:
                        cam.stop(); raise RuntimeError("dxcam no frame")
                    h, w = frame.shape[:2]
                    self.ctx = cam
                    self.region = (0, 0, w, h)
                    self.backend = "dxcam"
                    print(f"[grab] backend=dxcam region={self.region}")
                    return True
                elif b == "imagegrab":
                    img = ImageGrab.grab()  # RGB
                    w, h = img.size
                    self.backend = "imagegrab"
                    self.ctx = True
                    self.region = (0, 0, w, h)
                    print(f"[grab] backend=imagegrab region={self.region}")
                    return True
                elif b == "pyautogui":
                    img = pyautogui.screenshot()
                    w, h = img.size
                    self.backend = "pyautogui"
                    self.ctx = True
                    self.region = (0, 0, w, h)
                    print(f"[grab] backend=pyautogui region={self.region}")
                    return True
            except Exception as e:
                print(f"[grab] backend {b} failed: {e}")
                self.backend = None; self.ctx = None
        print("[grab] no working backend found")
        return False

    def stop(self):
        try:
            if self.backend == "mss" and self.ctx:
                self.ctx.close()
            elif self.backend == "dxcam" and self.ctx:
                self.ctx.stop()
        except Exception:
            pass
        finally:
            self.backend = None; self.ctx = None

    def grab_bgr(self):
        """Return BGR frame (numpy) or None."""
        if self.backend == "mss":
            shot = self.ctx.grab({"left": self.region[0], "top": self.region[1], "width": self.region[2], "height": self.region[3]})
            bgra = np.array(shot)
            return cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
        elif self.backend == "dxcam":
            frame = self.ctx.get_latest_frame()
            if frame is None:
                return None
            # dxcam -> RGB ndarray
            return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        elif self.backend == "imagegrab":
            rgb = np.array(ImageGrab.grab())
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        elif self.backend == "pyautogui":
            rgb = np.array(pyautogui.screenshot())
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return None


class ScreenShareFaceID:
    def __init__(self, window, video_label):
        self.window = window
        self.video_label = video_label
        self.running = False
        self.after_id = None
        self.grabber = _Grabber()
        self.known_encodings, self.known_names = load_known_faces()

    # совместимо с твоим main.py (ss.start())
    def start(self):
        if self.running:
            return
        if not self.grabber.start():
            print("[screenshare] no backend available")
            return
        self.running = True
        self._loop()

    def stop(self):
        self.running = False
        if self.after_id:
            try: self.window.after_cancel(self.after_id)
            except Exception: pass
            self.after_id = None
        self.grabber.stop()

    def _loop(self):
        if not self.running:
            return
        try:
            frame_bgr = self.grabber.grab_bgr()
            if frame_bgr is None or frame_bgr.size == 0:
                # показываем заглушку, чтобы видеть, что цикл живёт
                placeholder = np.full((300, 400, 3), 160, dtype=np.uint8)
                cv2.putText(placeholder, "No frame", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                out_rgb = cv2.cvtColor(placeholder, cv2.COLOR_BGR2RGB)
            else:
                # ресайз под окно (чтобы точно увидеть картинку)
                try:
                    ww = max(1, int(self.window.winfo_width()) - 40)
                    wh = max(1, int(self.window.winfo_height()) - 120)
                    if ww > 0 and wh > 0:
                        frame_bgr = cv2.resize(frame_bgr, (ww, wh), interpolation=cv2.INTER_AREA)
                except Exception:
                    pass

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                locs = face_recognition.face_locations(frame_rgb)
                encs = face_recognition.face_encodings(frame_rgb, locs)

                for (enc, (top, right, bottom, left)) in zip(encs, locs):
                    if self.known_encodings:
                        dist = face_recognition.face_distance(self.known_encodings, enc)
                        best_i = int(np.argmin(dist)); best_d = float(dist[best_i])
                        if best_d <= TOLERANCE:
                            name = self.known_names[best_i]; conf = confidence(best_d)
                        else:
                            name, conf = "Unknown", 0.0
                    else:
                        name, conf = "Unknown", 0.0

                    color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                    label = f"{name} {int(conf*100)}%" if name != "Unknown" else "Unknown"
                    cv2.rectangle(frame_bgr, (left, top), (right, bottom), color, 2)
                    (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    y1 = max(top - th - 6, 0)
                    cv2.rectangle(frame_bgr, (left, y1), (left + tw + 8, y1 + th + bl + 6), color, -1)
                    cv2.putText(frame_bgr, label, (left + 4, y1 + th + 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

                out_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            imgtk = ImageTk.PhotoImage(Image.fromarray(out_rgb))
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
        except Exception as e:
            print("[screenshare loop] error:", e)

        self.after_id = self.window.after(33, self._loop)  # ~30 fps
