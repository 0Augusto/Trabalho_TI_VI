import CNN as cnn

if __name__ == "__main__":
    cnn_test = cnn.CNN("./dataset/",250,250,32,14,6,"./dataset/output/model/")
    cnn_test.train()