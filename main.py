import tkinter as tk
from gui import AnomalyGUI

if __name__ == '__main__':
    root = tk.Tk()
    app = AnomalyGUI(root)
    root.mainloop()