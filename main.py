import tkinter as tk
import cv2
from Camera_Module import Camera
from yolo_detector import YoloDetector2
from face_recogmodule import ScreenShareFaceID
import time


window = tk.Tk()
window.geometry("1000x800")
window.title("Object Recognition App")
window.config(bg = "#A0A0A0")

def start_camera():
    window.geometry("1500x800")
    btn_screenshare.place_forget()
    button_start.place_forget()
    label_title.place_forget()
    btn_stop.place(relx = 0.5, rely = 0.7, anchor = "center")
    video_label.pack(pady=12)
    cam.start()

def stop_camera():
    window.geometry("1000x800")
    cam.stop()
    btn_stop.place_forget()
    video_label.pack_forget()
    btn_screenshare.place(relx = 0.6, rely = 0.7, anchor = "center")
    button_start.place(relx = 0.4, rely = 0.7, anchor = "center")
    label_title.place(relx=0.5, rely=0.2, anchor="center")

def on_close():
    cam.stop()
    window.destroy()

def start_the_screenshare():
    window.geometry("1000x800")
    stop_camera()
    ss.start()
    btn_screenshare.place_forget()
    button_start.place_forget()
    label_title.place_forget()
    btn_stop_screenshare.place(relx = 0.6, rely = 0.7, anchor = "center")
    video_label.pack(pady=12)

def stop_the_screenshare():
    window.geometry("1000x800")
    ss.stop()
    btn_stop_screenshare.place_forget()
    video_label.pack_forget()
    btn_screenshare.place(relx = 0.6, rely = 0.7, anchor = "center")
    button_start.place(relx = 0.4, rely = 0.7, anchor = "center")
    label_title.place(relx=0.5, rely=0.2, anchor="center")



button_start = tk.Button(window, text = "Start Camera", font = ("Comic Sans MS", 20), command = start_camera)
button_start.place(relx = 0.4, rely = 0.7, anchor = "center")

controls = tk.Frame(window)
controls.pack(pady = 8)

btn_stop = tk.Button(window, text = "Stop Camera", font = ("Comic Sans MS", 20), command = stop_camera)


btn_screenshare = tk.Button(window, text = "Screenshare", font  = ("Comic Sans MS", 20), command = start_the_screenshare)
btn_screenshare.place(relx = 0.6, rely = 0.7, anchor = "center")

btn_stop_screenshare = tk.Button(window, text = "Stop screenshare", font = ("Comic Sans MS", 20), command = stop_the_screenshare)

label_title = tk.Label(text = "Object And Face Recognition App", font = ("Comic Sans MS", 25), bg = "#A0A0A0")
label_title.place(relx = 0.5, rely = 0.2, anchor = "center")

video_label = tk.Label(window, relief = "solid")


yolo = YoloDetector2(weights = "yolov8n.pt", conf = 0.25, imgsz = 640)

def on_frame(frame_bgr):
    t0 = time.time()
    frame_bgr = yolo.draw_detections(frame_bgr)
    fps = 1.0 / max(1e-6, (time.time() - t0))
    cv2.putText(frame_bgr, f"FPS: {fps:1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
    return frame_bgr

cam = Camera(window, video_label, width = 900, height = 540, mirror = True, on_frame = on_frame)

ss = ScreenShareFaceID(window, video_label)

window.protocol("WM_DELETE_WINDOW", on_close)


window.mainloop()
