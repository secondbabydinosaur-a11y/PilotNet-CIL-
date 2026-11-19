"""ScreenVideo의 표시문 파일을 읽어 실제 화면 위에 현재 CIL 명령을 표시한다.

영상·센서·차량 제어에는 접근하지 않는다. 종료할 때 SIGTERM을 보내면 된다.
"""

import argparse
import json
import os
from pathlib import Path
import signal
import tempfile
import time
import tkinter as tk
from tkinter import font as tkfont

STEERING_INTERVENTION_LABEL = "STEERING INTERVENTION ACTIVE"


def fit_text(text, font, available_px, maximum_px, minimum_px):
    """폭에 맞춰 글자 크기를 줄이고, 그래도 길면 끝에 생략 기호를 붙인다."""
    for size in range(maximum_px, minimum_px - 1, -1):
        font.configure(size=-size)  # 음수 크기는 모니터 DPI와 무관한 픽셀 단위다.
        if font.measure(text) <= available_px:
            return text
    while text and font.measure(text + "…") > available_px:
        text = text[:-1]
    return text + "…"


def write_ready(path, value):
    """완성된 준비 상태를 원자적으로 공개해 부모 프로세스가 출발 전에 확인한다."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", prefix=".banner_ready_", dir=str(path.parent), delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream)
            stream.write("\n")
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


class LiveCommandBanner:
    """명령을 복사해 표시할 뿐, 명령 선택이나 차량 동작을 바꾸지 않는다."""

    def __init__(self, args):
        self.args = args
        self.stopping = False
        self.last_text = None
        self.ready = False
        self.root = tk.Tk()
        self.root.withdraw()  # 표시 속성을 설정하기 전에는 창을 노출하지 않는다.
        self.root.title("CIL live command")
        self.root.overrideredirect(True)  # 제목 표시줄이 없는 표시 전용 창이다.
        self.root.attributes("-topmost", True)
        try:
            self.root.attributes("-type", "notification")  # X11에서 포커스를 요구하지 않는 창이다.
        except tk.TclError:
            pass  # 해당 window-manager 힌트가 없어도 focus 요청은 하지 않는다.
        self.root.configure(background="#101827")
        self.root.geometry("{}x{}+{}+{}".format(args.width, args.height, args.x, args.y))
        self.main_font = tkfont.Font(root=self.root, family="DejaVu Sans", size=-36, weight="bold")
        self.detail_font = tkfont.Font(root=self.root, family="DejaVu Sans", size=-22)
        self.main = tk.Label(
            self.root, text="COMMAND: WAITING", font=self.main_font, fg="white",
            bg="#101827", anchor="w", borderwidth=0, takefocus=False,
        )
        self.main.pack(fill="x", padx=18, pady=(5, 0))
        self.detail = tk.Label(
            self.root, text="Waiting for command text…", font=self.detail_font, fg="#d5e5f5",
            bg="#101827", anchor="w", borderwidth=0, takefocus=False,
        )
        self.detail.pack(fill="x", padx=18, pady=(1, 5))
        self.root.protocol("WM_DELETE_WINDOW", self.request_stop)
        self.root.bind("<Escape>", self.request_stop)
        signal.signal(signal.SIGTERM, self.request_stop)
        signal.signal(signal.SIGINT, self.request_stop)
        self.refresh()  # 파일의 첫 유효 표시문을 읽고 나서 창을 표시한다.
        self.root.deiconify()
        self.root.after(0, self.mark_ready)

    def request_stop(self, *unused):
        self.stopping = True  # signal handler 안에서는 Tk 함수를 호출하지 않는다.

    def refresh(self):
        if self.stopping:
            self.root.destroy()
            return
        try:
            text = self.args.text_file.read_text(encoding="utf-8").replace("\x00", "")
            lines = text.splitlines()
            if not lines or not lines[0].strip():
                raise ValueError("command text is empty")
            main = lines[0]
            detail = " | ".join(line for line in lines[1:] if line.strip())
        except (OSError, UnicodeError, ValueError) as error:
            main = "COMMAND: WAITING"
            detail = "Command file unavailable: " + type(error).__name__
        intervention = main == STEERING_INTERVENTION_LABEL
        current = (main, detail, intervention)
        if current != self.last_text:
            available = max(100, self.args.width - 36)
            self.main.configure(
                text=fit_text(main, self.main_font, available, 40 if intervention else 36, 22),
                fg="#FFEA00" if intervention else "white",
                bg="#493700" if intervention else "#101827",
            )  # 실제 제어와 무관한 표시이며 개입이 끝나면 정상 색으로 즉시 복원한다.
            self.detail.configure(text=fit_text(detail, self.detail_font, available, 22, 14))
            self.last_text = current
        self.root.after(50, self.refresh)  # 같은 원본 파일을 약 20Hz로 읽는다.

    def mark_ready(self):
        if self.stopping or self.ready:
            return
        self.root.update_idletasks()
        if not self.root.winfo_ismapped():
            self.root.after(50, self.mark_ready)
            return
        if self.args.ready_file is not None:
            write_ready(self.args.ready_file, {
                "pid": os.getpid(), "title": "CIL live command", "mapped": True,
                "text_file": str(self.args.text_file), "x": self.args.x, "y": self.args.y,
                "width": self.args.width, "height": self.args.height, "ready_time": time.time(),
            })
        self.ready = True
        print("LIVE_COMMAND_BANNER_READY", flush=True)

    def run(self):
        self.root.mainloop()  # 차량 제어와 독립된 표시 프로세스다.


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text-file", type=Path, required=True)
    parser.add_argument("--x", type=int, required=True)
    parser.add_argument("--y", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, default=90)
    parser.add_argument("--ready-file", type=Path)
    args = parser.parse_args(argv)
    if args.x < 0 or args.y < 0 or args.width < 240 or args.height < 80:
        parser.error("x/y must be nonnegative, width >=240 and height >=80")
    args.text_file = args.text_file.expanduser().resolve()
    if args.ready_file is not None:
        args.ready_file = args.ready_file.expanduser().resolve()
        if args.ready_file.exists():
            parser.error("ready file already exists: " + str(args.ready_file))
    return args


def main():
    args = parse_args()
    try:
        banner = LiveCommandBanner(args)
        banner.run()
    except tk.TclError as error:
        raise SystemExit("live command banner could not use the X11/Tk display: " + str(error))


if __name__ == "__main__":
    main()
