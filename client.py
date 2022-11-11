import socket
import PIL as Image
import io
def client(image, host='localhost', port=8082): 

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    retorno = "" 
   
    server_address = (host, port) 
    print ("Conexão no host %s porta %s" % server_address) 
    sock.connect(server_address) 

    try: 

        message = "Enviando uma mensagem de teste" 
        print ("Enviando %s" % message)
   
        file = open(image,'rb')
        
        image_data = file.read(2048)

        while image_data:
            sock.send(image_data)
            image_data = file.read(2048)

        amount_received = 0 
        amount_expected = len(message)
        data = "" 

        while amount_received < amount_expected: 
            data += str(sock.recv(16))
            data = data.replace("b'", "")
            data = data.replace("'", "") 
            amount_received += len(data)
            retorno = "Recebendo: %s" %str(data) 
    except socket.error as e: 
        retorno = "Problema no Socket %s" %str(e) 
    except Exception as e: 
        retorno = "Exception: %s" %str(e) 
    finally: 
        print ("Conexão encerrada!") 
        sock.close()
        return retorno 
