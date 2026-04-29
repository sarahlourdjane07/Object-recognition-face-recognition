import cv2
from PIL import Image, ImageTk

class Camera:
    def __init__(self, window, video_label, width = 640, height = 480, mirror = True, on_frame = None):
        self.window = window
        self.video_label = video_label
        self.width = width
        self.height = height
        self.mirror = mirror
        self.on_frame = on_frame
        self.cap = None
        self.running = False
        self._after_id = None
        self._stride = 1
        self._counter = 0

    def set_on_frame(self, fn):
        self.on_frame = fn

    def set_processing_stirde(self, n:int):
        self._stride = max(1, int(n))


    def start(self):
        if self.running:
            return
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.video_label.config(text = "Cannot open camera")
            return
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        self.running = True
        self._loop()

    def stop(self):
        self.running = False
        if self._after_id is not None:
            try:
                self.window.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

    def _loop(self):
        if not self.running or self.cap is None:
            return

        ok, frame = self.cap.read()
        if not ok:
            self._after_id = self.window.after(10, self._loop)
            return

        if self.mirror:
            frame = cv2.flip(frame, 1)
        if callable(self.on_frame):
            self._counter = (self._counter + 1) % self._stride
            if self._counter == 0:
                try:
                    frame = self.on_frame(frame)
                except Exception:
                    pass
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        imgtk = ImageTk.PhotoImage(Image.fromarray(rgb))

        self.video_label.imgtk = imgtk
        self.video_label.configure(image = imgtk)

        self._after_id = self.window.after(10, self._loop)



