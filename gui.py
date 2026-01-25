import tkinter as tk
from tkinter import ttk, messagebox
import torch
import psutil
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import cv2
from PIL import Image, ImageTk
import glob
import os
import threading
import time
from utils import SmartConfig

from app_logic import AnomalyDetectionApp

class AnomalyGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("PatchCore + CNN Anomaly Classifier")
        self.master.geometry("1100x850")
        
        # zmienne od CUDA
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.setup_variables()
        
        # Glowna klasa GUI
        self.notebook = ttk.Notebook(self.master)
        self.notebook.pack(expand=True, fill="both", padx=10, pady=10)

        # inicjalizacja zakladek w gui
        self.page1 = ttk.Frame(self.notebook)
        self.page2 = ttk.Frame(self.notebook)
        self.page3 = ttk.Frame(self.notebook)
        self.page4 = ttk.Frame(self.notebook)
        self.page5 = ttk.Frame(self.notebook)
        self.page6 = ttk.Frame(self.notebook)

        self.notebook.add(self.page1, text="1. Ustawienia i Trening")
        self.notebook.add(self.page2, text="2. Macierz Pomyłek")
        self.notebook.add(self.page3, text="3. Testowanie PatchCore")
        self.notebook.add(self.page4, text="4. Testowanie PC+CNN")
        self.notebook.add(self.page5, text="5. Podgląd Augmentacji")
        self.notebook.add(self.page6, text="6. Wykresy")

        self.setup_status_bar()
        self.app_logic = AnomalyDetectionApp(self.device)
        self.setup_page1()
        self.setup_page2()
        self.setup_page3()
        self.setup_page4()
        self.setup_page5()
        self.setup_page6()
        
        # update co pare sekund zjadania zasobow (VRAM, CPU, RAM)
        self.update_resource_monitor()

        master.protocol("WM_DELETE_WINDOW", self.on_closing)

    def setup_variables(self):
        """ Inicjalizacja zmiennych Tkinter do przechowywania parametrów """
        # zmienna od monitorowania zuzycia
        self.monitoring_id = None
        # zmienne patchcore
        self.pc_threshold = tk.DoubleVar(value=0.79)
        self.pc_sampling = tk.DoubleVar(value=0.15)
        self.pc_status = tk.StringVar(value="Niezaładowany")
        self.status_pc = tk.StringVar(value="Status: Niezainicjalizowany")
        self.stats_results = {}
        # zmienne CNN
        self.cnn_img_size = tk.IntVar(value=128)
        self.cnn_batch_size = tk.IntVar(value=8)
        self.cnn_epochs = tk.IntVar(value=10)
        self.cnn_seed = tk.IntVar(value=5)
        self.crop_size = tk.IntVar(value=96)
        self.cnn_test_size = tk.DoubleVar(value=0.2)
        self.use_cropping = tk.BooleanVar(value=False)
        self.status_cnn = tk.StringVar(value="Status: Brak modelu")
        # zmienne od augmentacji
        self.aug_flips = tk.BooleanVar(value=False)
        self.aug_cutpaste = tk.BooleanVar(value=False)
        self.aug_dilation = tk.IntVar(value=5)
        self.aug_seg_ratio = tk.DoubleVar(value=190)
        self.aug_factor = tk.IntVar(value=1)

    def setup_status_bar(self):
        self.status_var = tk.StringVar(value="System gotowy")
        status_bar = ttk.Label(self.master, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    ###---PAGE 1---
    def setup_page1(self):
        """ Budowa pierwszej strony: Ustawienia """
        # sekcja patchcore
        pc_frame = ttk.LabelFrame(self.page1, text=" Ustawienia PatchCore ")
        pc_frame.pack(fill="x", padx=15, pady=10)

        ttk.Label(pc_frame, text="Anomaly Threshold:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        ttk.Entry(pc_frame, textvariable=self.pc_threshold, width=10).grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(pc_frame, text="Sampling Ratio (0.1 - 1.0):").grid(row=0, column=2, padx=5, pady=5, sticky="w")
        ttk.Entry(pc_frame, textvariable=self.pc_sampling, width=10).grid(row=0, column=3, padx=5, pady=5)

        self.btn_load_pc = ttk.Button(pc_frame, text="Wczytaj PatchCore do Pamięci", command=self.start_pc_loading_thread)
        self.btn_load_pc.grid(row=1, column=0, columnspan=2, padx=5, pady=10, sticky="ew")

        # status patchcore
        self.lbl_status_pc = ttk.Label(pc_frame, textvariable=self.status_pc, font=("Arial", 9, "italic"))
        self.lbl_status_pc.grid(row=2, column=0, columnspan=2, padx=5, pady=(0, 10))

        # sprawdzanie zuzycia zasobow
        self.lbl_resources = ttk.Label(pc_frame, text="Zasoby: CPU: 0% | RAM: 0% | VRAM: 0MB", foreground="blue")
        self.lbl_resources.grid(row=1, column=2, columnspan=2, padx=5, pady=5)

        # sekcja cnn
        cnn_frame = ttk.LabelFrame(self.page1, text=" Konfiguracja CNN ")
        cnn_frame.pack(fill="x", padx=15, pady=10)

        ttk.Label(cnn_frame, text="Image Size:").grid(row=0, column=0, padx=5, pady=5)
        ttk.Entry(cnn_frame, textvariable=self.cnn_img_size, width=10).grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(cnn_frame, text="Batch Size:").grid(row=0, column=2, padx=5, pady=5)
        ttk.Entry(cnn_frame, textvariable=self.cnn_batch_size, width=10).grid(row=0, column=3, padx=5, pady=5)

        ttk.Label(cnn_frame, text="Epochs:").grid(row=1, column=0, padx=5, pady=5)
        ttk.Entry(cnn_frame, textvariable=self.cnn_epochs, width=10).grid(row=1, column=1, padx=5, pady=5)
        ttk.Label(cnn_frame, text="Seed:").grid(row=0, column=4, padx=5, pady=5)
        ttk.Entry(cnn_frame, textvariable=self.cnn_seed, width=10).grid(row=0, column=5, padx=5, pady=5)
        ttk.Label(cnn_frame, text="Test size:").grid(row=0, column=6, padx=5, pady=5)
        ttk.Entry(cnn_frame, textvariable=self.cnn_test_size, width=10).grid(row=0, column=7, padx=5, pady=5)
        ttk.Label(cnn_frame, text="Crop size:").grid(row=0, column=8, padx=5, pady=5)
        ttk.Entry(cnn_frame, textvariable=self.crop_size, width=10).grid(row=0, column=9, padx=5, pady=5)
        ttk.Checkbutton(cnn_frame, text="Używaj wycinania anomalii (Crop)", variable=self.use_cropping).grid(row=1, column=2, columnspan=2, padx=5, pady=5)

        # wyswietlanie statusu cnn (np. epoka 5/15)
        self.lbl_status_cnn = ttk.Label(cnn_frame, textvariable=self.status_cnn, font=("Arial", 9, "bold"))
        self.lbl_status_cnn.grid(row=3, column=0, columnspan=4, padx=5, pady=(0, 10))

        # przyicski od CNN
        self.btn_train = ttk.Button(cnn_frame, text="Uruchom trening CNN", command=self.start_cnn_training_thread)
        self.btn_train.grid(row=2, column=0, columnspan=2, padx=5, pady=10, sticky="ew")

        btn_save = ttk.Button(cnn_frame, text="Zapisz wytrenowany Model")
        btn_save.grid(row=2, column=2, columnspan=2, padx=5, pady=10, sticky="ew")

        btn_load = ttk.Button(cnn_frame, text="Wczytaj wytrenowany Model")
        btn_load.grid(row=2, column=4, columnspan=2, padx=5, pady=10, sticky="ew")
        # augmentacja danych
        aug_frame = ttk.LabelFrame(self.page1, text=" Augmentacja danych ")
        aug_frame.pack(fill="x", padx=15, pady=10)

        ttk.Checkbutton(aug_frame, text="Włącz Flips + Rotations", variable=self.aug_flips).grid(row=0, column=0, padx=10, pady=5, sticky="w")
        ttk.Checkbutton(aug_frame, text="Włącz Cut-and-Paste", variable=self.aug_cutpaste).grid(row=1, column=0, padx=10, pady=5, sticky="w")

        ttk.Label(aug_frame, text="Dilation Kernel Size:").grid(row=0, column=1, padx=5, pady=5, sticky="w")
        ttk.Entry(aug_frame, textvariable=self.aug_dilation, width=8).grid(row=0, column=2, padx=5, pady=5)

        ttk.Label(aug_frame, text="Segm. Threshold Ratio:").grid(row=1, column=1, padx=5, pady=5, sticky="w")
        ttk.Entry(aug_frame, textvariable=self.aug_seg_ratio, width=8).grid(row=1, column=2, padx=5, pady=5)
        ttk.Label(aug_frame, text="Mnożnik augmentacji:").grid(row=2, column=1, padx=5, pady=5, sticky="w")

        # spinbox do zadawania jaki jest mnoznik augmentacji
        tk.Spinbox(
            aug_frame, 
            from_=1, 
            to=10, 
            textvariable=self.aug_factor, 
            width=5
        ).grid(row=2, column=2, padx=5, pady=5, sticky="w")

        ttk.Label(aug_frame, textvariable=self.aug_factor).grid(row=2, column=3, padx=5, pady=5)
    def start_pc_loading_thread(self):
        """ Uruchamia proces wczytywania w tle """
        self.btn_load_pc.config(state="disabled") # blokowanie przycisku
        ratio = self.pc_sampling.get()
        
        thread = threading.Thread(target=self.run_pc_loading, args=(ratio,))
        thread.daemon = True # watek sie zamyka wraz zakonczeniem apki
        thread.start()

    def run_pc_loading(self, ratio):
        """ Metoda wykonywana w wątku boczny """
        def update_ui(msg, is_error=False):
            self.status_var.set(msg)
            if is_error:
                messagebox.showerror("Błąd PatchCore", msg)
            if "gotowy" in msg.lower() or is_error:
                self.btn_load_pc.config(state="normal")
                self.pc_status.set("Załadowany" if not is_error else "Błąd")

        success = self.app_logic.load_patchcore_to_memory(
            sampling_ratio=ratio,
            status_callback=update_ui
        )
        if success:
            self.status_pc.set("Status: GOTOWY (Załadowano)")
            self.lbl_status_pc.config(foreground="green")
    def start_cnn_training_thread(self):
        """ Uruchamia proces przygotowania danych i treningu CNN w osobnym wątku. """
        # blokujemy przycisk treningu zeby nie odpalic 2x przypadkowo
        self.btn_train.config(state="disabled")
        
        def status_callback(epoch, total_epochs, loss, acc):
            """ Funkcja wywoływana przez app_logic po każdej epoce """
            self.status_cnn.set(f"Status: Epoka {epoch}/{total_epochs} | Loss: {loss} | Acc: {acc}%")
            self.master.update_idletasks()

        def run_training():
            try:
                # pobieramy parametry z gui
                config = SmartConfig({
                    'epochs': self.cnn_epochs.get(),
                    'batch_size': self.cnn_batch_size.get(),
                    'img_size': self.cnn_img_size.get(),
                    'seed': self.cnn_seed.get(),
                    'test_size': self.cnn_test_size.get(),
                    'use_cropping': self.use_cropping.get(),
                    'aug_dilation': self.aug_dilation.get(),
                    'aug_seg_ratio': self.aug_seg_ratio.get(),
                    'aug_cutpaste' : self.aug_cutpaste.get(),
                    'aug_factor' : self.aug_factor.get(),
                    'aug_flips': self.aug_flips.get(),
                    'crop_size' : self.crop_size.get()
                })

                self.status_cnn.set("Status: Przygotowanie danych...")
                
                # zwracamy historie do wykresow
                history = self.app_logic.train_pipeline(config, status_callback)

                # informujemy ze sie trening skonczyl
                self.status_cnn.set("Status: Trening zakończony!")
                self.lbl_status_cnn.config(foreground="green")
                
                # aktualizuejym wykresy w zakladce nr6
                if hasattr(self, 'update_plots'):
                    self.update_plots(history)
                
                messagebox.showinfo("Sukces", "Model CNN został wytrenowany i jest gotowy.")

            except Exception as e:
                self.status_cnn.set(f"Status: BŁĄD TRENINGU")
                self.lbl_status_cnn.config(foreground="red")
                messagebox.showerror("Błąd Treningu", str(e))
            finally:
                self.btn_train.config(state="normal")

        # uruchamiamy na osobnym watku trenowanie
        thread = threading.Thread(target=run_training, daemon=True)
        thread.start()
    ###---END PAGE 1---
    ###---PAGE 2---
    def setup_page2(self):
        """ Budowa drugiej strony: Macierz Pomyłek """
        ctrl_frame = ttk.Frame(self.page2)
        ctrl_frame.pack(side=tk.TOP, fill="x", padx=10, pady=10)

        self.btn_calc_cm = ttk.Button(ctrl_frame, text="Oblicz macierz pomyłek", command=self.generate_confusion_matrix)
        self.btn_calc_cm.pack(side=tk.LEFT, padx=5)

        ttk.Label(ctrl_frame, text="Status:").pack(side=tk.LEFT, padx=20)
        self.cm_status_lbl = ttk.Label(ctrl_frame, text="Oczekiwanie na dane...", foreground="gray")
        self.cm_status_lbl.pack(side=tk.LEFT)

        # dolny panel z wykresem
        self.plot_container = ttk.Frame(self.page2)
        self.plot_container.pack(side=tk.TOP, expand=True, fill="both", padx=10, pady=10)
        
        # placeholdery na macierz
        self.cm_placeholder = ttk.Label(self.plot_container, text="Tu pojawi się macierz po zakończeniu treningu i kliknięciu przycisku.")
        self.cm_placeholder.place(relx=0.5, rely=0.5, anchor="center")
    def generate_confusion_matrix(self):
        """
        Pobiera dane testowe z app_logic, wykonuje predykcję modelem CNN 
        i wyświetla macierz pomyłek.
        """
      
        if self.app_logic.classifier is None:
            messagebox.showerror("Błąd", "Model CNN nie jest załadowany! Wytrenuj model lub wczytaj go z pliku.")
            return

        # czyscimy wykres
        for widget in self.plot_container.winfo_children():
            widget.destroy()

        self.cm_status_lbl.config(text="Generowanie predykcji...", foreground="blue")
        self.master.update_idletasks()

        try:
            y_true = []
            y_pred = []

            self.app_logic.classifier.eval()
            device = self.app_logic.device

            # pobieramy rozmiar obrazu z pola gui (self.crop_size lub self.img_size)
            # w zaleznosci od tego czy uzywamy wycinania anomalii
            if self.app_logic.model_trained_with_crops:
                size = int(self.crop_size.get())
            else:
                size = int(self.cnn_img_size.get())

            # pobieramy transformacje o odpowiednim rozmiarze
            transform = self.app_logic.get_transform(size)

            # predykcja na probkach testowych
            # self.app.test_samples ma liste (PIL_Image, label_idx)
            if not self.app_logic.test_samples:
                raise Exception("Brak próbek testowych. Przeprowadź najpierw trening lub załaduj dane.")

            with torch.no_grad():
                for img_pil, label_idx in self.app_logic.test_samples:
                    # img_pil jest przygotowanym obrazem (crop lub resized 512)
                    # zmieniamy na tensor  o rozmiarze wejściowym CNN
                    img_tensor = transform(img_pil).unsqueeze(0).to(device)

                    outputs = self.app_logic.classifier(img_tensor)
                    _, preds = torch.max(outputs, 1)

                    y_true.append(label_idx)
                    y_pred.append(preds.item())

            # tworzymy wykres confusion matrixa
            labels = self.app_logic.anomaly_classes
            fig, ax = plt.subplots(figsize=(8, 6))

            # generujemy macierz z funkcji
            cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))



            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=labels, yticklabels=labels, ax=ax)

            ax.set_title("Macierz Pomyłek - Klasyfikator CNN")
            ax.set_xlabel("Przewidziana klasa")
            ax.set_ylabel("Rzeczywista klasa")
            fig.tight_layout()

            # dajemy wykres do tkintera
            canvas = FigureCanvasTkAgg(fig, master=self.plot_container)
            canvas_widget = canvas.get_tk_widget()
            canvas_widget.pack(expand=True, fill="both")
            canvas.draw()

            self.cm_status_lbl.config(text="Wygenerowano pomyślnie", foreground="green")

        except Exception as e:
            self.cm_status_lbl.config(text=f"Błąd: {str(e)}", foreground="red")
            messagebox.showerror("Błąd", f"Nie udało się wygenerować macierzy: {e}")
    ###---END PAGE 2---
    
    #---PAGE 3---
    def setup_page3(self):
        """ Strona 3: Testowanie PatchCore z podziałem na klasy """
        self.p3_main_frame = ttk.Frame(self.page3)
        self.p3_main_frame.pack(expand=True, fill="both")

        # nowy zagniezdzony notebook w ktorym trzymamy klasy anomalii
        self.class_notebook = ttk.Notebook(self.p3_main_frame)
        self.class_notebook.pack(side=tk.LEFT, expand=True, fill="both", padx=5, pady=5)

        # panel boczny z wynikami po prawej
        self.res_panel = ttk.LabelFrame(self.p3_main_frame, text=" Wynik PatchCore ", width=300)
        self.res_panel.pack(side=tk.RIGHT, fill="y", padx=10, pady=10)
        self.res_panel.pack_propagate(False) # stala szerokosc panelu

        self.setup_results_ui()
        
        # ladujemy dynamicznie klasy
        self.load_test_folders("./dataset/test")
    
    def load_test_folders(self, base_path):
        """ Skanuje folder test i tworzy zakładki """
        if not os.path.exists(base_path):
            ttk.Label(self.class_notebook, text=f"Nie znaleziono ścieżki: {base_path}").pack()
            return

        self.test_data = {} # { 'klasa': [lista_sciezek] }
        folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]

        for folder in folders:
            path = os.path.join(base_path, folder)
            images = glob.glob(os.path.join(path, "*.png")) + glob.glob(os.path.join(path, "*.jpg"))
            if images:
                self.test_data[folder] = sorted(images)
                
                # tworzymy zakladke dla danej klasy
                tab = ttk.Frame(self.class_notebook)
                self.class_notebook.add(tab, text=folder.upper())
                
                # obszar na obraz
                canvas_label = ttk.Label(tab) # tu wyswietlamy obraz
                canvas_label.pack(expand=True, fill="both")
                tab.image_label = canvas_label # trzebazapamietac referencje

        # inicjalizacja indeksow
        self.current_class = folders[0] if folders else None
        self.current_img_idx = 0
        
        # bindujemy zakladki
        self.class_notebook.bind("<<NotebookTabChanged>>", self.on_tab_changed)

    def on_tab_changed(self, event):
        """ Wywoływane przy kliknięciu w zakładkę klasy """
        tab_id = self.class_notebook.select()
        self.current_class = self.class_notebook.tab(tab_id, "text").lower()
        self.current_img_idx = 0
        self.display_current_test_image()
        if hasattr(self, 'lbl_p4_counter'):
            self.change_image_p4(0)

    def change_image(self, delta):
        """ Przewijanie zdjęć w bieżącej klasie """
        if not self.current_class or self.current_class not in self.test_data:
            return
        
        new_idx = self.current_img_idx + delta
        if 0 <= new_idx < len(self.test_data[self.current_class]):
            self.current_img_idx = new_idx
            self.display_current_test_image()

    def display_current_test_image(self):
        """ Wyświetla tylko czysty obraz (bez analizy) """
        if not self.current_class: return
        
        img_path = self.test_data[self.current_class][self.current_img_idx]
        
        # otwieranie zdjecia
        img_pil = Image.open(img_path).convert("RGB").resize((600, 600))
        img_tk = ImageTk.PhotoImage(img_pil)

        current_tab_idx = self.class_notebook.index("current")
        tab = self.class_notebook.nametowidget(self.class_notebook.tabs()[current_tab_idx])
        tab.image_label.config(image=img_tk)
        tab.image_label.image = img_tk
        
        # resetujemy etykiety dopoki nie klikniemy 'zbadaj'
        self.lbl_score.config(text="Score: ---", foreground="black")
        self.lbl_eval.config(text="Oczekiwanie...", foreground="black")
    def setup_results_ui(self):
        """ Elementy panelu wyników po prawej stronie """
        self.lbl_score = ttk.Label(self.res_panel, text="Anomaly Score: ---", font=("Arial", 12))
        self.lbl_score.pack(pady=5)

        self.lbl_time = ttk.Label(self.res_panel, text="Czas analizy: ---", foreground="blue")
        self.lbl_time.pack(pady=5)

        self.lbl_eval = ttk.Label(self.res_panel, text="Ocena: ---", font=("Arial", 14, "bold"))
        self.lbl_eval.pack(pady=10)
        
        stats_frame = ttk.LabelFrame(self.res_panel, text=" Statystyki Klasyfikacji ")
        stats_frame.pack(fill="x", padx=5, pady=10)
        
        self.lbl_batch_status = ttk.Label(stats_frame, text="Status: Gotowy", foreground="blue")
        self.lbl_batch_status.pack(pady=2)

        self.txt_stats = tk.Text(stats_frame, height=20, width=35, font=("Consolas", 9))
        self.txt_stats.pack(padx=5, pady=5)
        
        # przycisk do analizy wszystkich klas
        ttk.Button(self.res_panel, text="Analizuj wszystkie klasy", 
                   command=self.start_batch_analysis_thread).pack(pady=5, fill="x")
        # Nprzycisk do badania tylko obecnego zdjecia
        self.btn_analyze = ttk.Button(self.res_panel, text="Zbadaj zdjęcie", command=self.start_analysis_thread)
        self.btn_analyze.pack(pady=10)

        nav_frame = ttk.Frame(self.res_panel)
        nav_frame.pack(side=tk.BOTTOM, pady=20)
        
        ttk.Button(nav_frame, text="< Poprzedni", command=lambda: self.change_image(-1)).grid(row=0, column=0, padx=5)
        ttk.Button(nav_frame, text="Następny >", command=lambda: self.change_image(1)).grid(row=0, column=1, padx=5)
    def update_resource_monitor(self):
        cpu = psutil.cpu_percent()
        ram = psutil.virtual_memory().percent
        vram_text = "VRAM: N/A"
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 1024**2
            reserved = torch.cuda.memory_reserved() / 1024**2
            vram_text = f"ALLOC: {alloc:.0f} MB | RESERVED: {reserved:.0f} MB"

        self.lbl_resources.config(text=f"Zasoby: CPU: {cpu}% | RAM: {ram}% | {vram_text}")
        # trzeba zapisac id zadania
        self.monitoring_id = self.master.after(2000, self.update_resource_monitor)
    def start_analysis_thread(self):
        """ Uruchamia analizę PatchCore w tle """
        if not self.app_logic.patchcore_ready:
            messagebox.showwarning("Uwaga", "Najpierw wczytaj PatchCore na Stronie 1!")
            return

        img_path = self.test_data[self.current_class][self.current_img_idx]
        self.btn_analyze.config(state="disabled") # blokujemy przycisk na czas obliczen
        self.status_var.set("Trwa analiza obrazu...")

        threading.Thread(target=self._run_inference_task, args=(img_path,), daemon=True).start()

    def _run_inference_task(self, img_path):
        """ Zadanie wykonywane przez wątek roboczy """
        try:
            # operacja na gpu do predykcji
            start_time = time.time()
            anomaly_map, score = self.app_logic.get_patchcore_prediction(img_path)
            end_time = time.time()
            duration = end_time - start_time
            # po obliczeniach wracamy do glownego watku z wynikami
            self.after_idle_call(self.update_ui_with_results, anomaly_map, score, img_path, duration)
        except Exception as e:
            self.after_idle_call(self.status_var.set, f"Błąd analizy: {e}")
            self.after_idle_call(self.btn_analyze.config, {"state": "normal"})

    def update_ui_with_results(self, anomaly_map, score, img_path, duration):
        """ Aktualizuje obraz o heatmapę w głównym wątku """
        # wczytujemy oryginalne zdjecie
        img_bgr = cv2.imread(img_path)
        img_bgr = cv2.resize(img_bgr, (512, 512))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # generujemy heatmape do zdjecia
        am_min, am_max = anomaly_map.min(), anomaly_map.max()
        norm_map = ((anomaly_map - am_min) / (am_max - am_min + 1e-5) * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(norm_map, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        display_img = cv2.addWeighted(img_rgb, 0.6, heatmap_rgb, 0.4, 0)
        
        # trzeba zaktualizowac etykiety
        self.lbl_score.config(text=f"Anomaly Score: {score:.2f}", 
                              foreground="red" if score > self.pc_threshold.get() else "green")
        self.lbl_eval.config(text="ANOMALIA" if score > self.pc_threshold.get() else "OK")
        self.lbl_time.config(text=f"Czas analizy: {duration:.3f} s")

        # wyswietlamy obraz
        img_pil = Image.fromarray(display_img).resize((600, 600))
        img_tk = ImageTk.PhotoImage(img_pil)
        
        tab_id = self.class_notebook.select()
        tab = self.class_notebook.nametowidget(tab_id)
        tab.image_label.config(image=img_tk)
        tab.image_label.image = img_tk
        
        self.btn_analyze.config(state="normal")
        self.status_var.set(f"Analiza zakończona. Score: {score:.2f}")
    def start_batch_analysis_thread(self):
        if not self.app_logic.patchcore_ready:
            messagebox.showwarning("Błąd", "Najpierw wczytaj PatchCore!")
            return
        threading.Thread(target=self.run_batch_analysis, daemon=True).start()

    def run_batch_analysis(self):
        self.stats_results = {}
        threshold = self.pc_threshold.get()
        
        # iterujemy po klasach z  load_test_folders
        for class_name, images in self.test_data.items():
            self.stats_results[class_name] = {'anomalia': 0, 'ok': 0, 'total': len(images)}
            
            for i, img_path in enumerate(images):
                # aktualizujemy status w ui
                self.lbl_batch_status.config(text=f"Badanie: {class_name.upper()} {i+1}/{len(images)}")
                
                # robimy predykcje patchcore
                _, score = self.app_logic.get_patchcore_prediction(img_path)
                
                if score > threshold:
                    self.stats_results[class_name]['anomalia'] += 1
                else:
                    self.stats_results[class_name]['ok'] += 1
                
                # odswiezamy na zywo status 
                self.refresh_stats_display()

        self.lbl_batch_status.config(text="Status: Analiza zakończona", foreground="green")

    def refresh_stats_display(self):
        """ Odświeża okno tekstowe ze statystykami """
        self.txt_stats.delete('1.0', tk.END)
        report = ""
        for cls, s in self.stats_results.items():
            report += f"{cls.upper()}:\n"
            report += f"  Anomalie: {s['anomalia']}/{s['total']}\n"
            report += f"  Dobre:    {s['ok']}/{s['total']}\n"
            report += "-" * 20 + "\n"
        
        self.txt_stats.insert(tk.END, report)
        self.master.update_idletasks()
    #---END PAGE3---
    #--- PAGE 4---
    def setup_page4(self):
        """ Strona 4: Profesjonalny Dashboard Hybrydowy """
        main_f = ttk.Frame(self.page4)
        main_f.pack(expand=True, fill="both", padx=10, pady=10)

        header = ttk.Label(main_f, text="Analiza Hybrydowa: PatchCore (Lokalizacja) + CNN (Klasyfikacja)", font=("Arial", 14, "bold"))
        header.pack(pady=5)

        # kontener na 3 kolumny
        cols_f = ttk.Frame(main_f)
        cols_f.pack(expand=True, fill="both")

        # 1 kolumna - oryginal
        col1 = ttk.LabelFrame(cols_f, text=" Oryginalny Obraz ")
        col1.pack(side=tk.LEFT, expand=True, fill="both", padx=5)
        self.p4_orig_label = ttk.Label(col1)
        self.p4_orig_label.pack(expand=True)

        # 2 kolumna - wynik detekcji
        col2 = ttk.LabelFrame(cols_f, text=" Detekcja Anomalii (PatchCore) ")
        col2.pack(side=tk.LEFT, expand=True, fill="both", padx=5)
        self.p4_det_label = ttk.Label(col2)
        self.p4_det_label.pack(expand=True)

        # 3 kolumna - panel sterowania i wynik
        col3 = ttk.Frame(cols_f, width=250)
        col3.pack(side=tk.LEFT, fill="y", padx=10)

        # podglad wycinki ktory szedlby do CNNa
        crop_f = ttk.LabelFrame(col3, text=" Wycinek dla CNN ")
        crop_f.pack(fill="x", pady=5)
        self.p4_crop_label = ttk.Label(crop_f)
        self.p4_crop_label.pack(pady=5)

        # wyniki
        res_f = ttk.LabelFrame(col3, text=" Wyniki Klasyfikacji ")
        res_f.pack(fill="x", pady=5)
        self.lbl_p4_status = ttk.Label(res_f, text="Status: Oczekiwanie", foreground="orange")
        self.lbl_p4_status.pack(pady=5)
        self.lbl_p4_class = ttk.Label(res_f, text="KLASA: ---", font=("Arial", 12, "bold"))
        self.lbl_p4_class.pack(pady=10)

        # nawigacja
        nav_f = ttk.LabelFrame(col3, text=" Nawigacja ")
        nav_f.pack(fill="x", side=tk.BOTTOM, pady=10)
        
        self.lbl_p4_counter = ttk.Label(nav_f, text="Zdjęcie: 0/0", font=("Arial", 9, "bold"))
        self.lbl_p4_counter.pack(pady=2)

        btn_frame = ttk.Frame(nav_f) # kontener z przyciskami
        btn_frame.pack(fill="x")

        ttk.Button(btn_frame, text="<", width=5, command=lambda: self.change_image_p4(-1)).pack(side=tk.LEFT, expand=True, padx=2)
        ttk.Button(btn_frame, text="ANALIZUJ", command=self.run_hybrid_analysis, style="Accent.TButton").pack(side=tk.LEFT, expand=True, padx=2)
        ttk.Button(btn_frame, text=">", width=5, command=lambda: self.change_image_p4(1)).pack(side=tk.LEFT, expand=True, padx=2)

    def change_image_p4(self, step):
        """ Przewijanie zdjęć na stronie 4 z aktualizacją licznika """
        if not self.current_class or self.current_class not in self.test_data:
            return
        
        # obliczamy nowy indeks
        total_imgs = len(self.test_data[self.current_class])
        new_idx = self.current_img_idx + step

        if new_idx >= total_imgs: new_idx = 0
        if new_idx < 0: new_idx = total_imgs - 1

        self.current_img_idx = new_idx
        img_path = self.test_data[self.current_class][self.current_img_idx]

        # aktualizujemy licznik i status
        self.lbl_p4_counter.config(text=f"Zdjęcie: {self.current_img_idx + 1} / {total_imgs}")
        self.lbl_p4_status.config(text=f"Klasa: {self.current_class.upper()}", foreground="blue")

        # ladujemy podglad oryginalu
        img_pil = Image.open(img_path).convert("RGB").resize((400, 400))
        img_tk = ImageTk.PhotoImage(img_pil)
        self.p4_orig_label.config(image=img_tk)
        self.p4_orig_label.image = img_tk

        # czyscimy poprzednie wyniki
        self.p4_det_label.config(image='')
        self.p4_crop_label.config(image='')
        self.lbl_p4_class.config(text="KLASA: ---")

    def run_hybrid_analysis(self):
        """ Ulepszona analiza z wizualizacją krok po kroku """
        if not self.app_logic.patchcore_ready or not self.app_logic.classifier:
            messagebox.showwarning("Błąd", "Wymagany załadowany PatchCore ORAZ Model CNN!")
            return

        self.lbl_p4_status.config(text="Status: Analizowanie...", foreground="red")
        self.master.update_idletasks()

        img_path = self.test_data[self.current_class][self.current_img_idx]
        img_pil = Image.open(img_path).convert("RGB").resize((512, 512))
        
        # lokalizujemy anomalie z patchcorem
        amap, score = self.app_logic.get_patchcore_prediction(img_path)
        
        # wycinamy fragment do CNNa
        crop_pil, (x1, y1, x2, y2) = self.app_logic.get_anomaly_crop(img_pil, amap)
        
        # pokazujemy wycinek w malym oknie
        crop_display = ImageTk.PhotoImage(crop_pil.resize((150, 150)))
        self.p4_crop_label.config(image=crop_display)
        self.p4_crop_label.image = crop_display
        
        # klasyfikacja cnn
        target_size = self.crop_size.get() if self.app_logic.model_trained_with_crops else self.cnn_img_size.get()

        img_size = self.cnn_img_size.get() if hasattr(self.cnn_img_size, 'get') else self.cnn_img_size

        target_size = target_size if self.app_logic.model_trained_with_crops else img_size

        crop_tensor = self.app_logic.get_transform(target_size)(crop_pil).unsqueeze(0).to(self.device)
        self.app_logic.classifier.eval()
        with torch.no_grad():
            out = self.app_logic.classifier(crop_tensor)
            prob = torch.nn.functional.softmax(out, dim=1)
            conf, pred = torch.max(prob, 1)
            class_name = self.app_logic.anomaly_classes[pred.item()]

        img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        label_txt = f"{class_name.upper()} ({conf.item()*100:.1f}%)"
        cv2.rectangle(img_cv, (x1, y1-30), (x1+200, y1), (0, 255, 0), -1)
        cv2.putText(img_cv, label_txt, (x1+5, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # wyswietlamy wynik detekcji
        res_img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)).resize((400, 400))
        res_img_tk = ImageTk.PhotoImage(res_img_pil)
        self.p4_det_label.config(image=res_img_tk)
        self.p4_det_label.image = res_img_tk

        # aktualizacja tekstu
        self.lbl_p4_class.config(text=f"KLASA: {class_name.upper()}")
        self.lbl_p4_status.config(text="Status: Gotowe", foreground="green")
    #---END PAGE4---
    #---PAGE 5---
    def setup_page5(self):
        """ Strona 5: Podgląd Augmentacji """
        self.p5_container = ttk.Frame(self.page5)
        self.p5_container.pack(expand=True, fill="both")

        # 2 panele obok siebie
        self.lbl_orig = ttk.Label(self.p5_container, text="Oryginał")
        self.lbl_orig.grid(row=0, column=0, padx=10, pady=10)
        
        self.lbl_aug = ttk.Label(self.p5_container, text="Po augmentacji / wycięciu")
        self.lbl_aug.grid(row=0, column=1, padx=10, pady=10)

        ttk.Button(self.page5, text="Generuj losowy podgląd", command=self.preview_augmentation).pack(pady=10)

    def preview_augmentation(self):
        # wybieramy losowe probki z folderow anomalii
        print("Użyto preview augmentation.")
        print(f" Wartośc aug_cutpaste {self.aug_cutpaste.get()} oraz patchcore_ready {self.app_logic.patchcore_ready}" )
        import random
        all_classes = [c for c in self.test_data.keys() if c != 'good']
        cls = random.choice(all_classes)
        path = random.choice(self.test_data[cls])
        
        img_pil = Image.open(path).convert("RGB").resize((512, 512))
        
        # wyswietlamy orginal z lewej strony
        tk_orig = ImageTk.PhotoImage(img_pil.resize((400, 400)))
        self.lbl_orig.config(image=tk_orig)
        self.lbl_orig.image = tk_orig

        res_img = img_pil
        info = "Oryginał"

        if self.aug_cutpaste.get() and self.app_logic.patchcore_ready:
            good_path = random.choice(self.test_data.get('good', []))
            target_pil = Image.open(good_path).convert("RGB").resize((512,512))
            
            # pobieramy mape anomalii dla obecnego zdjecia z path
            amap, _ = self.app_logic.get_patchcore_prediction(path)
            
            from utils import apply_cut_and_paste
            aug_threshold = self.aug_seg_ratio.get()
            aug_dilation = self.aug_dilation.get()
            print(f"Threshold: {aug_threshold}, Dilation: {aug_dilation}" )
            res_img = apply_cut_and_paste(img_pil, target_pil, amap, aug_threshold, aug_dilation)
            info = "Cut-and-Paste: Anomalia na zdrowym orzechu"

        if self.aug_flips.get():
            res_img = res_img.transpose(Image.FLIP_LEFT_RIGHT)
            info += " + Flip"

        # wyswietlamy wynik
        tk_aug = ImageTk.PhotoImage(res_img.resize((400, 400)))
        self.lbl_aug.config(image=tk_aug, text=info)
        self.lbl_aug.image = tk_aug
    #---END PAGE5---
    #---PAGE 6---
    def setup_page6(self):
        """ Strona 6: Statystyki Treningu """
        self.p6_frame = ttk.Frame(self.page6)
        self.p6_frame.pack(expand=True, fill="both")

        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(10, 4))
        self.canvas_plot = FigureCanvasTkAgg(self.fig, master=self.p6_frame)
        self.canvas_plot.get_tk_widget().pack(expand=True, fill="both")

        ttk.Button(self.p6_frame, text="Odśwież Wykresy", command=self.update_plots).pack(pady=10)
    def update_plots(self, history=None):
        if history is None: return

        self.ax1.clear()
        self.ax1.plot(history['train_loss'], label='Loss', color='red')
        self.ax1.plot(history['val_loss'], label='Val Loss')
        self.ax1.set_title("Krzywa uczenia (Błąd)")
        self.ax1.set_xlabel("Epoch")
        self.ax1.legend()

        self.ax2.clear()
        self.ax2.plot(history['val_acc'], label='Val Accuracy', color='green')
        self.ax2.plot(history['train_acc'], label='Train Acc', color='blue')
        self.ax2.set_title("Accuracy (%)")
        self.ax2.set_xlabel("Epoch")
        self.ax2.legend()

        self.fig.tight_layout()
        self.canvas_plot.draw()
    #---END PAGE 6---
    def on_closing(self):
        if messagebox.askokcancel("Wyjście", "Czy chcesz zamknąć program?"):
            if self.monitoring_id:
                self.master.after_cancel(self.monitoring_id)
            self.master.destroy()

    def after_idle_call(self, func, *args):

        self.master.after_idle(lambda: func(*args))

    