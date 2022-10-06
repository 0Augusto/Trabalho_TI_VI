import tkinter as tk
from tkinter import *
from tkinter import filedialog
from tkinter import simpledialog
from PIL import ImageTk, Image
import re
import sys

interface = Tk()
interface.title("FaceRecon")#Nome do programa
#tamanho em pixels da janela inicial
interface.geometry("800x600")
interface.resizable(False, False)
#backGround (bg) color
interface['bg'] = "#aaaaaa"
#=====//=====//=====//=====//=====//=====//=====
# define options for opening or saving a file
file_opt = options = {}
options['defaultextension'] = '.jpeg, .png'
options['initialfile'] = '*.jpeg, *.png'

class Operacoes:
    
    def importImage(self):
        filename = filedialog.askopenfilename()
        global image
        if filename:
            image = Image.open(filename)
            image = image.resize((250,250))
            image = ImageTk.PhotoImage(image)
            interface.janelaImagem.create_image(200, 200, image=image)
            interface.textoMsg.delete('1.0', tk.END)           
            interface.textoMsg.insert('1.0', "Imagem carregada com sucesso")
        else:
            interface.textoMsg.delete('1.0', tk.END)           
            interface.textoMsg.insert('1.0', "Erro ao carregar a imagem")
    def classificador(self):
        interface.textoMsg.delete('1.0', tk.END)           
        interface.textoMsg.insert('1.0', "Funcionalidade em desenvolvimento")

#Interface do menu. Cada botão executa uma tarefa
class Application(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        operacoes = Operacoes()

        frameBotoes = Frame(interface, width=800, height=30)
        frameBotoes.grid(row=0, column=0)

        adicionar = Button(frameBotoes, text="Adicionar imagem", padx=10, pady=10, border=2, command=operacoes.importImage)
        adicionar.grid(row=0, column=0)
        
        btnTeste1 = Button(frameBotoes, text="Classificador", padx=10, pady=10, border=2, command=operacoes.classificador)
        btnTeste1.grid(row=0, column=1)
        #btnTeste2 = Button(frameBotoes, text="Teste 2", padx=10, pady=10, border=2)
        #btnTeste2.grid(row=0, column=1)
        #btnTeste3 = Button(frameBotoes, text="Teste 3", padx=10, pady=10, border=2)
        #btnTeste3.grid(row=0, column=2)
        #btnTeste4 = Button(frameBotoes, text="Teste 4", padx=10, pady=10, border=2)
        #btnTeste4.grid(row=0, column=3)
        self.janelaImagem = Canvas(interface, width=400, height=600, bg="#999999")
        self.janelaImagem.grid(row=1, column=0)
        janelaMsg = Frame(interface, width=400, height=600, bg="#eeeeee")
        janelaMsg.grid(row=1, column=1)
        self.textoMsg = Text(janelaMsg)
        self.textoMsg.grid(row=0,column=0)

        #adicionar.pack()
        #self.pack()

interface = Application()

#executar sempre a aplicação, para que a aplicação não encerre
interface.mainloop()

# from PIL import ImageTk, Image

# self.image_area = Canvas(self, width=300, height=300, bg="#C8C8C8")
# self.image_area. grid,place ou pack dependendo

# def openImage(self):
   # pip install Pillow