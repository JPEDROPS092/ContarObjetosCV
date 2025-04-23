import tkinter as tk
from tkinter import ttk

def main():
    print("Iniciando teste do Tkinter...")
    root = tk.Tk()
    root.title("Teste Tkinter")
    root.geometry("300x200")
    
    label = ttk.Label(root, text="Se você está vendo esta janela, o Tkinter está funcionando!")
    label.pack(pady=20)
    
    button = ttk.Button(root, text="Fechar", command=root.destroy)
    button.pack(pady=10)
    
    print("Mostrando janela...")
    root.mainloop()
    print("Teste concluído.")

if __name__ == "__main__":
    main() 