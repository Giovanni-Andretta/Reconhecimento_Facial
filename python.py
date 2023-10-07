# https://www.youtube.com/watch?v=SEDwevQ0wic

#Giovanni Andretta Carbonero
#Bruno Lopes dos Reis

import cv2
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
import time
import getpass
from hashlib import sha512

user = []
secret = []

def face_detection(test_img):
    gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    face_haar_cascade = cv2.CascadeClassifier('face_detection/haarcascades/haarcascade_frontalface_default.xml')
    faces = face_haar_cascade.detectMultiScale(gray_img, scaleFactor=1.3, minNeighbors=5)
    return faces,gray_img

def train_classifier(faces, face_ID):
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(faces, np.array(faceid))
    return face_recognizer

def draw_rect(test_img, face):
    (x, y, w, h) = face
    cv2.rectangle(test_img, (x, y), (x + w, y + h), (0, 255, 0), thickness = 3)
    
def put_text(test_img, text, x, y):
    cv2.putText(test_img, text, (x, y), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 1)
    
def labels_for_training_data(directory):
    faces = []
    faceid = []
    
    for path, subdirnames, filenames in os.walk(directory):
        for filename in filenames:
            if filename.startswith("."):
                print("Skypping system files")
                continue
            id = os.path.basename(path)
            img_path = os.path.join(path, filename)
            print("Verificando...")
            test_img = cv2.imread(img_path)
            if test_img is None:
                print("Image not loaded properly")
                continue
            faces_rect, gray_img = face_detection(test_img)
            if len(faces_rect) != 1:
                continue
            (x, y, w, h) = faces_rect[0]
            rol_gray = gray_img[y : y + w, x : x + h]
            faces.append(rol_gray)
            faceid.append(int(id))
        return faces, faceid

def valid():
    global faceid, confidence
    confidence = 0
    
    test_img = cv2.imread('temp/temp.jpg')
    faces_detected, gray_img = face_detection(test_img)
    print("face detected",faces_detected)

    faces, faceid = labels_for_training_data('face/0')
    face_recognizer = train_classifier(faces, faceid)
    face_recognizer.save('trainingdata.yml')
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.read('trainingdata.yml')
    name={0 : user[0]}

    for face in faces_detected:
        (x, y, w, h) = face
        rol_gray = gray_img[y : y + h, x : x + h]
        label,confidence = face_recognizer.predict(rol_gray)
        print("Confidence", confidence)
        print("Label", label)
        draw_rect(test_img, face)
        predict_name = name[label]
        
        #Verifica se é ou não a pessoa e coloca o nome
        if confidence > 68:
            continue
        put_text(test_img, predict_name, x, y)

    for (x, y, w, h) in faces_detected:
        cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 0, 0), thickness = 1)
        pass
    
    resized_img = cv2.resize(test_img, (1000, 700))
    cv2.imshow("Face Detection", resized_img)
    
    #Temporizador da janela de validacao
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    #Verifica se é ou não a pessoa autorizada
    if confidence > 68 or confidence == 0:
        opt = str(input("\n\n\nRosto não identificado. \n\nDeseja fazer a validação por usuario e senha? ('S' para sim): "))
        opt = opt.upper()
        opt = opt[0]
        
        if opt == "S":
            login()
        else:
            sys.exit()
    else:
        print("\n\n\t\tBem vindo sr(a). " + user[0] + "\n")
        
def login():
    opt = 3
    while True:
        print("\n\n\t\tLogin")
        usuario = str(input("\nDigite seu usuario: "))
        senha = str(getpass.getpass("Digite sua senha: "))
        cripto = criptografia(senha)

        if usuario in user[:]:
            if cripto in secret[:]:
                print("\n\n\t\tBem vindo sr(a). " + user[0] + "\n")
                break
            else:
                opt-=1
                print("\nUsuario ou senha incorretos!\n\nTente novamente... tentativas restantes: " + str(opt))
        else:
            opt-=1
            print("\nUsuario ou senha incorretos!\n\nTente novamente... tentativas restantes: " + str(opt))

        if opt == 0:
            print("\n\nFechando programa.")
            sys.exit()


def temp():
    camera_port = 0
    camera = cv2.VideoCapture(camera_port)
    file = "temp/temp.jpg"

    print ("Digite <ESC> para sair / <s> para Salvar")

    if camera.isOpened():
        ret, frame = camera.read()
        #check whether frame is successfully captured
        if ret:
            # continue to display window until 'q' is pressed
            while(True):
                ret, frame = camera.read()
                cv2.imshow("Camera", frame)
                k = cv2.waitKey(100)
                if k == 27:
                    break
                elif k == ord('s'):
                    cv2.imwrite(file,frame)
                    break
            cv2.destroyAllWindows()
            camera.release()
        #print error if frame capturing was unsuccessful
        else:
            print("Error : Failed to capture frame")
    # print error if the connection with camera is unsuccessful
    else:
        print("Cannot open camera")

    return k


def face():
    camera = 0
    camera = cv2.VideoCapture(camera)
    aux = user[0]
    i = 0

    while(True):
        ret, frame = camera.read()
        cv2.imshow("Camera", frame)
        cv2.imwrite("face/0/" + str(aux) + "_" + str(i) + ".jpg", frame)
        print("Mude de posicao")
        cv2.waitKey(2000)
        i+=1
        if i == 11:
            break
    cv2.destroyAllWindows()
    camera.release()
    
def preenchendo_valores():
    with open('secrets/usuario.txt', 'r') as arquivo_u:
        for valor in arquivo_u:
            user.append(valor)
            
    with open('secrets/senha.txt', 'r') as arquivo_s:
        for valor in arquivo_s:
            secret.append(valor)

def criptografia(senha):
    hash_senha = sha512(senha.encode())
    hash_senha = hash_senha.hexdigest()
    return hash_senha

def cadastro():
    with open('secrets/usuario.txt', 'w') as arquivo_u:
        usuario = str(input("Digite seu nome: "))
        arquivo_u.write(str(usuario))
        
    with open('secrets/senha.txt', 'w') as arquivo_s:
        senha = str(getpass.getpass("Digite sua senha: "))
        cripto = criptografia(senha)
        arquivo_s.write(str(cripto))
        

def inic():    
    with open('secrets/usuario.txt') as arquivo:
        arquivo.seek(0)
        first_char = arquivo.read(1)
        if not first_char:
            cadastro()
            preenchendo_valores()
            face()
            main()
        else:
            arquivo.seek(0)
            preenchendo_valores()
            if temp() != 27:
                valid()
            else:
                sys.exit()
            
def menu():
    while True:
        print("="*50)
        print("\t\tMenu do Editor")
        print("\nEscolha sua opção:")
        print("\n\n[1] - Editor de fotos")
        print("[0] - Sair\n")

        print("="*50)
    
        aux = int(input("Digite sua opção: "))

        if aux == 0:
            sys.exit()
        elif aux == 1:
            edit()
        else:
            print("\nOpcao invalida!\n\nTente novamente...\n")
            time.sleep(2)
    
def edit():
    print("="*50)
    print("\t\tEditor de fotos")
    opt = str(input("\n\nDigite o Path: "))
    image = cv2.imread(opt)
    
    print("\n")
    print("="*50)
    
    print("\t\tFoto Cinza\n")
    imagecinza = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imagemCinza = cv2.cvtColor(imagecinza, cv2.IMREAD_GRAYSCALE)
    plt.imshow(imagemCinza)
    plt.show()
    
    print("\n")
    print("="*50)
    
    print("\t\tFoto Negativa\n")
    img_neg = -1 * image + 255
    plt.figure(figsize=(5,5))
    plt.imshow(img_neg, cmap = "gray")
    plt.show()
    
    print("\n")
    print("="*50)
    
    print("\t\tHistograma\n")
    hist = cv2.calcHist([image],[0],None,[256],[0,256])
    valorintenso = np.array([x for x in range(hist.shape[0])])
    plt.bar(valorintenso, hist[:,0], width = 5)
    plt.show()
    
def main():
    inic()
    menu()
    
main()
