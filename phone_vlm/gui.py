"""Небольшое локальное окно Tkinter без веб-сервера.

Нейросеть работает в одном рабочем потоке. Обращения к Tkinter выполняет
только основной поток, получая ответы через очередь. Поэтому интерфейс
не зависает во время загрузки модели и не обучает её при распознавании.
"""

from __future__ import annotations

import queue
import threading
from pathlib import Path

from .common import ProjectError
from .inference import Recognizer, append_log


def launch(run_dir: Path, config: dict, project_root: Path, device: str, log: Path) -> None:
    import tkinter as tk
    from tkinter import filedialog, ttk
    from PIL import Image, ImageTk

    window = tk.Tk()
    window.title("Распознавание смартфонов — MobileCLIP2-S0")
    window.geometry("920x650")
    window.minsize(720, 580)
    window.columnconfigure(0, weight=1)
    window.rowconfigure(2, weight=1)
    tasks, events = queue.Queue(), queue.Queue()
    selected = [None]
    recognizer = [None]
    preview = [None]  # Ссылка не даёт Tkinter удалить картинку до перерисовки.

    status = tk.StringVar(value="Загрузка модели…")
    filename = tk.StringVar(value="Файл ещё не выбран")
    answer = tk.StringVar(value="")
    warnings = tk.StringVar(value="Оценка уверенности не является гарантией правильности.")
    toolbar = ttk.Frame(window, padding=12)
    toolbar.grid(row=0, column=0, sticky="ew")
    toolbar.columnconfigure(2, weight=1)
    choose = ttk.Button(toolbar, text="Выбрать фотографию", state="disabled")
    choose.grid(row=0, column=0, padx=(0, 10))
    process = ttk.Button(toolbar, text="Распознать", state="disabled")
    process.grid(row=0, column=1)
    ttk.Label(window, textvariable=filename, padding=(12, 0), wraplength=860).grid(row=1, column=0, sticky="w")
    picture = ttk.Label(window, anchor="center")
    picture.grid(row=2, column=0, sticky="nsew", padx=12, pady=10)
    ttk.Label(window, textvariable=answer, font=("Arial", 16), wraplength=860,
              padding=(12, 4)).grid(row=3, column=0, sticky="w")
    ttk.Label(window, textvariable=warnings, wraplength=860,
              padding=(12, 4)).grid(row=4, column=0, sticky="w")
    ttk.Label(window, textvariable=status, wraplength=860,
              padding=12).grid(row=5, column=0, sticky="w")

    def worker():
        try:
            recognizer[0] = Recognizer(run_dir, config, project_root, device)
            events.put(("ready", recognizer[0].metadata["model_version"]))
        except Exception as exc:
            events.put(("load_error", str(exc)))
            return
        while True:
            path = tasks.get()
            if path is None:
                return
            try:
                result = recognizer[0].predict(path)
                append_log(log, result)
                events.put(("result", result))
            except Exception as exc:
                # Даже неожиданная ошибка отдельного запроса не убивает поток.
                events.put(("request_error", str(exc)))

    def select_file():
        path = filedialog.askopenfilename(title="Выберите фотографию",
                    filetypes=[("Фотографии", "*.jpg *.jpeg *.png"), ("Все файлы", "*.*")])
        if not path:
            return
        selected[0] = Path(path)
        filename.set(selected[0].name)
        answer.set("")
        warnings.set("")
        try:
            # Проверяем ограничения ДО декодирования превью большого файла.
            from .images import read_rgb
            image = read_rgb(selected[0], recognizer[0].metadata["image_policy"])
            image.thumbnail((820, 350), Image.Resampling.BILINEAR)
            preview[0] = ImageTk.PhotoImage(image)
            picture.configure(image=preview[0])
            process.configure(state="normal")
            status.set("Можно запускать распознавание.")
        except (ProjectError, OSError) as exc:
            picture.configure(image="")
            preview[0] = None
            process.configure(state="disabled")
            status.set(str(exc))

    def start_prediction():
        choose.configure(state="disabled")
        process.configure(state="disabled")
        status.set("Обработка…")
        tasks.put(selected[0])

    def poll():
        try:
            while True:
                kind, data = events.get_nowait()
                if kind == "ready":
                    choose.configure(state="normal")
                    status.set(f"Модель {data} загружена. Выберите фотографию.")
                elif kind == "load_error":
                    status.set("Не удалось загрузить модель: " + data)
                else:
                    choose.configure(state="normal")
                    process.configure(state="normal" if selected[0] else "disabled")
                    if kind == "request_error":
                        status.set(data)
                    elif data["status"] == "error":
                        answer.set("Распознавание не выполнено")
                        status.set(data["message"])
                    else:
                        answer.set(f"{data['answer']} — оценка уверенности {data['confidence_percent']:.2f}%")
                        warnings.set("\n".join(data["warnings"]) or "Оценка уверенности не калибрована.")
                        status.set(f"Модель {data['model_version']}; {data['elapsed_seconds']:.3f} с. Результат записан в журнал.")
        except queue.Empty:
            pass
        window.after(100, poll)

    choose.configure(command=select_file)
    process.configure(command=start_prediction)
    threading.Thread(target=worker, daemon=True).start()
    window.after(100, poll)
    window.protocol("WM_DELETE_WINDOW", lambda: (tasks.put(None), window.destroy()))
    window.mainloop()
