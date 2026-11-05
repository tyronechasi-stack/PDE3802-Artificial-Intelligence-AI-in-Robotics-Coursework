"""Desktop app: classify photos/webcam, run the pipeline, and view test results.
Run: python simple_app.py  (put it next to the other scripts)"""

import json
import queue
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

import cv2
from PIL import Image, ImageTk

from predict_image import classify_frame, load_classifier

ROOT = Path(__file__).resolve().parent
TEST_OUT = ROOT / "outputs/test_results"
MAX_SIZE = (640, 480)


def set_text(widget, text):
    widget.config(state="normal")
    widget.delete("1.0", "end")
    widget.insert("1.0", text)
    widget.config(state="disabled")


class App:
    def __init__(self, root):
        self.root = root
        root.title("Item Classifier")
        self.model = self.classes = self.transform = None
        self.camera = None
        self.running = False
        self.lines = queue.Queue()

        self.tabs = ttk.Notebook(root)
        self.tabs.pack(fill="both", expand=True, padx=8, pady=8)
        self.build_classify_tab()
        self.build_train_tab()
        self.build_results_tab()
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.load_model()
        self.refresh_results()

    # ---------- model ----------
    def load_model(self):
        try:
            self.model, self.classes, self.transform = load_classifier()
            self.result.config(text="Open a photo or start the camera")
            return True
        except (FileNotFoundError, ValueError) as error:
            self.model = None
            self.result.config(text=f"No model yet: {error}")
            return False

    # ---------- classify tab ----------
    def build_classify_tab(self):
        tab = ttk.Frame(self.tabs)
        self.tabs.add(tab, text="Classify")
        self.video = tk.Label(tab, width=80, height=24, bg="black")
        self.video.pack(padx=10, pady=10)
        self.result = tk.Label(tab, text="", font=("Arial", 16), wraplength=620)
        self.result.pack(pady=5)
        buttons = ttk.Frame(tab)
        buttons.pack(pady=10)
        ttk.Button(buttons, text="Open photo", command=self.open_photo).pack(side="left", padx=5)
        self.camera_button = ttk.Button(buttons, text="Start camera", command=self.toggle_camera)
        self.camera_button.pack(side="left", padx=5)

    def show(self, frame):
        index, confidence = classify_frame(frame, self.model, self.transform)
        name = self.classes[index].split("_", 1)[1].replace("_", " ")
        self.result.config(text=f"{name}: {confidence:.1%}")
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image.thumbnail(MAX_SIZE)
        photo = ImageTk.PhotoImage(image)
        self.video.config(image=photo, width=photo.width(), height=photo.height())
        self.video.image = photo  # keep a reference

    def need_model(self):
        if self.model is None and not self.load_model():
            messagebox.showwarning("No model", "Train a model first (Train tab).")
            return False
        return True

    def open_photo(self):
        self.stop_camera()
        if not self.need_model():
            return
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png")])
        if not path:
            return
        frame = cv2.imread(path)
        if frame is None:
            messagebox.showerror("Error", "Cannot read that image.")
            return
        self.show(frame)

    def toggle_camera(self):
        if self.camera is not None:
            self.stop_camera()
            return
        if not self.need_model():
            return
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            camera.release()
            messagebox.showerror("Error", "Cannot open camera. Check permissions.")
            return
        self.camera = camera
        self.camera_button.config(text="Stop camera")
        self.update_camera()

    def update_camera(self):
        if self.camera is None:
            return
        ok, frame = self.camera.read()
        if not ok:
            self.stop_camera()
            messagebox.showerror("Error", "The camera stopped supplying frames.")
            return
        self.show(frame)
        self.root.after(30, self.update_camera)

    def stop_camera(self):
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        self.camera_button.config(text="Start camera")

    # ---------- train tab ----------
    def build_train_tab(self):
        tab = ttk.Frame(self.tabs)
        self.tabs.add(tab, text="Train")
        bar = ttk.Frame(tab)
        bar.pack(fill="x", padx=10, pady=10)
        self.epochs = tk.IntVar(value=10)
        self.step_buttons = [
            ttk.Button(bar, text="1. Prepare data",
                       command=lambda: self.run_script("prepare_data.py")),
            ttk.Button(bar, text="2. Train",
                       command=lambda: self.run_script("train_model.py", "--epochs", str(self.epochs.get()))),
            ttk.Button(bar, text="3. Test",
                       command=lambda: self.run_script("test_model.py")),
        ]
        self.step_buttons[0].pack(side="left", padx=4)
        self.step_buttons[1].pack(side="left", padx=4)
        ttk.Label(bar, text="Epochs:").pack(side="left", padx=(10, 2))
        ttk.Spinbox(bar, from_=1, to=50, width=5, textvariable=self.epochs).pack(side="left")
        self.step_buttons[2].pack(side="left", padx=14)
        self.log = ScrolledText(tab, height=24, width=90, state="disabled", font=("Courier", 10))
        self.log.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        set_text(self.log, "Run the steps in order. Output appears here.\n"
                           "Photos go in data/raw/<class>/<item_group>/ first.")

    def run_script(self, script, *args):
        if self.running:
            return
        try:
            int(self.epochs.get())
        except (tk.TclError, ValueError):
            messagebox.showerror("Error", "Epochs must be a whole number.")
            return
        self.stop_camera()
        self.running = True
        for button in self.step_buttons:
            button.config(state="disabled")
        set_text(self.log, f"Running {script}...\n")
        threading.Thread(target=self.worker, args=(script, args), daemon=True).start()
        self.poll(script)

    def worker(self, script, args):
        code = 1
        try:
            process = subprocess.Popen(
                [sys.executable, "-u", script, *args], cwd=ROOT,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in process.stdout:
                self.lines.put(line)
            process.wait()
            code = process.returncode
        except OSError as error:
            self.lines.put(f"{error}\n")
        self.lines.put(("done", code))

    def poll(self, script):
        finished = None
        text = []
        while not self.lines.empty():
            item = self.lines.get()
            if isinstance(item, tuple):
                finished = item[1]
            else:
                text.append(item)
        if text:
            self.log.config(state="normal")
            self.log.insert("end", "".join(text))
            self.log.see("end")
            self.log.config(state="disabled")
        if finished is None:
            self.root.after(100, self.poll, script)
        else:
            self.finish(script, finished)

    def finish(self, script, code):
        self.running = False
        for button in self.step_buttons:
            button.config(state="normal")
        if code != 0:
            messagebox.showerror("Failed", f"{script} failed. See the log for details.")
            return
        if script == "train_model.py":
            self.load_model()
        elif script == "test_model.py":
            self.refresh_results()
            self.tabs.select(self.results_tab)
        messagebox.showinfo("Done", f"{script} finished.")

    # ---------- results tab ----------
    def build_results_tab(self):
        tab = self.results_tab = ttk.Frame(self.tabs)
        self.tabs.add(tab, text="Test results")
        top = ttk.Frame(tab)
        top.pack(fill="x", padx=10, pady=8)
        self.metrics = tk.Label(top, font=("Arial", 14), anchor="w", justify="left")
        self.metrics.pack(side="left")
        ttk.Button(top, text="Refresh", command=self.refresh_results).pack(side="right")
        middle = ttk.Frame(tab)
        middle.pack(fill="both", expand=True, padx=10)
        self.matrix = tk.Label(middle)
        self.matrix.pack(side="left", padx=(0, 10))
        self.report = tk.Text(middle, width=55, height=16, state="disabled", font=("Courier", 10))
        self.report.pack(side="left", fill="both", expand=True)
        ttk.Label(tab, text="Error analysis").pack(anchor="w", padx=10, pady=(8, 0))
        self.errors = ScrolledText(tab, height=8, state="disabled", font=("Courier", 10))
        self.errors.pack(fill="x", padx=10, pady=(0, 10))

    def refresh_results(self):
        metrics_file = TEST_OUT / "metrics.json"
        if not metrics_file.exists():
            self.metrics.config(text="No test results yet. Run step 3 in the Train tab.")
            self.matrix.config(image="")
            for widget in (self.report, self.errors):
                set_text(widget, "")
            return
        try:
            data = json.loads(metrics_file.read_text(encoding="utf-8"))
            self.metrics.config(
                text=f"Test images: {data['test_images']}    "
                     f"Accuracy: {data['accuracy']:.3f}    Macro-F1: {data['macro_f1']:.3f}")
            report = TEST_OUT / "class_report.txt"
            set_text(self.report, report.read_text(encoding="utf-8") if report.exists() else "")
            errors = TEST_OUT / "error_analysis.md"
            set_text(self.errors, errors.read_text(encoding="utf-8") if errors.exists() else "")
            picture = TEST_OUT / "confusion_matrix.png"
            if picture.exists():
                with Image.open(picture) as image:
                    image.thumbnail((480, 420))
                    photo = ImageTk.PhotoImage(image.copy())
                self.matrix.config(image=photo)
                self.matrix.image = photo
        except (OSError, ValueError, KeyError) as error:
            self.metrics.config(text=f"Cannot read test results: {error}")

    def close(self):
        self.stop_camera()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()