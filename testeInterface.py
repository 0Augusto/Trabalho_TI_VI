from tkinter import *
from tkinter import filedialog
from tkinter import simpledialog
from PIL import ImageTk, Image
import re
import sys

interface = Tk()
interface.title("FaceRecon")#Nome do programa
#tamanho em pixels da janela inicial
interface.geometry("1500x1080")
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
            interface.janelaImagem.create_image(450, 300, image=image)

#Interface do menu. Cada botão executa uma tarefa
class Application(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        operacoes = Operacoes()

        adicionar = Button(interface, text="Adicionar imagem", padx=10, pady=10, border=2, command=operacoes.importImage)
        adicionar.grid(row=0, column=0)
        frameBotoes = Frame(interface, width=600, height=30)
        frameBotoes.grid(row=0, column=1)
        btnTeste1 = Button(frameBotoes, text="Classificador", padx=10, pady=10, border=2)
        btnTeste1.grid(row=0, column=0)
        #btnTeste2 = Button(frameBotoes, text="Teste 2", padx=10, pady=10, border=2)
        #btnTeste2.grid(row=0, column=1)
        #btnTeste3 = Button(frameBotoes, text="Teste 3", padx=10, pady=10, border=2)
        #btnTeste3.grid(row=0, column=2)
        #btnTeste4 = Button(frameBotoes, text="Teste 4", padx=10, pady=10, border=2)
        #btnTeste4.grid(row=0, column=3)
        self.janelaImagem = Canvas(interface, width=900, height=720, bg="#999999")
        self.janelaImagem.grid(row=1, column=0)
        janelaMsg = Frame(interface, width=500, height=720, bg="#eeeeee")
        janelaMsg.grid(row=1, column=1)
        textoMsg = Text(janelaMsg, height=12)
        textoMsg['state'] = 'disabled'
        textoMsg.insert('1.0', "Olá, aqui vai aparecer a mensagem que tiver que aparecer :)")
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