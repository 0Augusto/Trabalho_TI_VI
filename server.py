import socket
import CNN as Cnn

def server(host = 'localhost', port=8082):
    data_payload = 2048

    sock = socket.socket(socket.AF_INET,  socket.SOCK_STREAM)
    
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
 
    server_address = (host, port)
    print ("Inicializando servidor %s porta %s" % server_address)
    
    sock.bind(server_address)
   
    sock.listen(5) 
    i = 0
    while True: 
        print ("Aguardando a mensagem do cliente")
        client, address = sock.accept() 
        data = client.recv(data_payload) 
        if data:
            cnn = Cnn("./dataset/",250,250,32,14,6,"./dataset/output/model/")
            # Converter de string para um caminho no servidor
            result = cnn.predict_single_img(data.decode())
            print ("Valor: %s" %data)
            client.send(result.encode())  # Tentei passar o resultado da predição assim, não testei, mas imagino q de certo
            print ("Enviando dado %s para o host %s" % (data, address))
            # end connection
            client.close()
            i+=1
            if i>=3: break           
server()