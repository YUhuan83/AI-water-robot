import tkinter as tk
from tkinter import ttk
import time
import threading
import winsound

WORK_TIME = 25 * 60
BREAK_TIME = 5 * 60
LONG_BREAK_TIME = 15 * 60
POMOS_BEFORE_LONG_BREAK = 4

class PomodoroTimer:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("番茄钟")
        self.root.geometry("320x300")
        self.root.resizable(False, False)
        self.root.configure(bg="#f0f0f0")

        self.state = "idle"
        self.remaining = WORK_TIME
        self.pomo_count = 0
        self.is_break = False
        self.running = False
        self.timer_id = None

        self.setup_ui()
        self.update_display()
        self.root.mainloop()

    def setup_ui(self):
        self.root.columnconfigure(0, weight=1)

        title = tk.Label(self.root, text="🍅 番茄钟", font=("Segoe UI", 18, "bold"), bg="#f0f0f0", fg="#333")
        title.pack(pady=(15, 5))

        self.timer_label = tk.Label(self.root, text="25:00", font=("Segoe UI", 48, "bold"), bg="#f0f0f0", fg="#e74c3c")
        self.timer_label.pack(pady=5)

        self.status_label = tk.Label(self.root, text="准备开始", font=("Segoe UI", 11), bg="#f0f0f0", fg="#888")
        self.status_label.pack()

        self.pomo_label = tk.Label(self.root, text="已完成: 0 个番茄", font=("Segoe UI", 10), bg="#f0f0f0", fg="#aaa")
        self.pomo_label.pack(pady=(5, 10))

        btn_frame = tk.Frame(self.root, bg="#f0f0f0")
        btn_frame.pack(pady=5)

        self.start_btn = tk.Button(btn_frame, text="开始", width=8, font=("Segoe UI", 10),
                                    command=self.toggle_timer, bg="#2ecc71", fg="white",
                                    relief="flat", cursor="hand2", activebackground="#27ae60")
        self.start_btn.pack(side="left", padx=5)

        self.reset_btn = tk.Button(btn_frame, text="重置", width=8, font=("Segoe UI", 10),
                                    command=self.reset, bg="#95a5a6", fg="white",
                                    relief="flat", cursor="hand2", activebackground="#7f8c8d")
        self.reset_btn.pack(side="left", padx=5)

    def toggle_timer(self):
        if self.running:
            self.pause()
        else:
            self.start()

    def start(self):
        if self.state == "idle":
            self.state = "work"
            self.remaining = WORK_TIME
        self.running = True
        self.start_btn.config(text="暂停", bg="#e67e22", activebackground="#d35400")
        self.tick()

    def pause(self):
        self.running = False
        if self.timer_id:
            self.root.after_cancel(self.timer_id)
            self.timer_id = None
        self.start_btn.config(text="继续", bg="#2ecc71", activebackground="#27ae60")

    def reset(self):
        self.running = False
        if self.timer_id:
            self.root.after_cancel(self.timer_id)
            self.timer_id = None
        self.state = "idle"
        self.remaining = WORK_TIME
        self.is_break = False
        self.start_btn.config(text="开始", bg="#2ecc71", activebackground="#27ae60")
        self.status_label.config(text="准备开始", fg="#888")
        self.update_display()

    def tick(self):
        if not self.running:
            return
        self.update_display()
        if self.remaining <= 0:
            self.on_complete()
            return
        self.remaining -= 1
        self.timer_id = self.root.after(1000, self.tick)

    def on_complete(self):
        self.running = False
        self.start_btn.config(text="开始", bg="#2ecc71", activebackground="#27ae60")
        winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
        for _ in range(3):
            winsound.MessageBeep(winsound.MB_OK)
            time.sleep(0.3)

        if not self.is_break:
            self.pomo_count += 1
            self.pomo_label.config(text=f"已完成: {self.pomo_count} 个番茄")
            if self.pomo_count % POMOS_BEFORE_LONG_BREAK == 0:
                self.remaining = LONG_BREAK_TIME
                self.is_break = True
                self.status_label.config(text="🎉 长休息时间！", fg="#9b59b6")
            else:
                self.remaining = BREAK_TIME
                self.is_break = True
                self.status_label.config(text="☕ 休息一下", fg="#3498db")
        else:
            self.remaining = WORK_TIME
            self.is_break = False
            self.state = "work"
            self.status_label.config(text="专注工作中", fg="#e74c3c")
            self.start()

        self.update_display()
        if not self.is_break:
            self.start()

    def format_time(self, secs):
        m, s = divmod(secs, 60)
        return f"{m:02d}:{s:02d}"

    def update_display(self):
        self.timer_label.config(text=self.format_time(self.remaining))
        if self.is_break:
            color = "#3498db" if self.remaining > BREAK_TIME else "#2ecc71"
        elif self.state == "work":
            color = "#e74c3c"
        else:
            color = "#e74c3c"
        self.timer_label.config(fg=color)

if __name__ == "__main__":
    PomodoroTimer()
