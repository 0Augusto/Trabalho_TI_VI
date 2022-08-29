# TI-VI
Trabalho combinando as disciplinas Computação Distribuída, Computação Paralela e Processamento e Análise de Imagem.


  Sprint 0: Início da discussão sobre o trabalho e, qual caminho seguir para o reconhecimento de imagens, paralelização e sobre computação distribuída. Das três opções que o grupo escolheu, detecção da espécie de ursos, distância das galáxias em relação ao planeta terra e, renconhecimento facila. O grupo optou pelo reconhecimento facial, e o modo de paralelizar escolhido foi, identificar múltiplas faces simultaneamente, ao que se encontra a computação distribuída, decidimos por um cliente-servidor que, o cliente faz o requerimento ao servidor e o último retorna a(s) face(s) pesquisadas pelo cliente, a base de dados escolhida pelo grupo foi LFW - People (Face Recognition)(https://www.kaggle.com/datasets/atulanandjha/lfwpeople).
  Ela contém mais de 13 mil imagens de rostos coletadas na web, cada face foi destinada uma etiqueta com o nome da pessoa referente a face, 1680 faces das mais de 13 mil recolhidas possuem duas ou mais fotos distintas no banco de dados. A única constante é que, os rostos foram identificados pelo framework Viola-Jones, o qual não é uma CNN.
  É uma coleção de JPEG do website (http://vis-www.cs.umass.edu/lfw/)cada rosto famoso e cada pixel (RGB) e codificado em flutuante em um raio de 0.0-1.0.
  
  Sprint 1: Apresentação do projeto, base de dados, cronograma (interface gráfica funcional para a sprint 2).
  
  Sprint 2: Revisão breve do projeto, apresentação da Sprint, apresentar a interface gráfica funcinoal com a implementação (parcial) do algoritmo (CNN) para o reconhecimento de faces, reconhecimento de múltiplas faces implementação inicial, cronograma: finalizar o projeto para a Sprint 3.
  
  Sprint 3: Revisão breve do projeto, apresentação da Sprint até o momento, interface gráfica totalmente funcional juntamente com o algoritmo CNN reconhecimento multiplas faces (paralelização) requisitadas pelo cliente e sendo retornado pelo servidor (computação paralela), apresentar acurácia e precisão do método escolhido.
