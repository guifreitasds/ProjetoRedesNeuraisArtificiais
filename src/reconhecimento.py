import cv2
from deepface import DeepFace

# Lista de imagens conhecidas e nomes associados
known_face_images = ["eugui.jpeg"]
known_face_names = ["Guilherme Freitas"]
#model_name = "VGG-Face"
#model_name = "OpenFace"
#model_name = "DeepID"
model_name = "Facenet512"

# Carregar imagens e calcular embeddings com DeepFace
known_face_embeddings = []
for image_path in known_face_images:
    embedding = DeepFace.represent(img_path=image_path, model_name=model_name)[0]["embedding"]
    known_face_embeddings.append(embedding)

# Iniciar captura de vídeo
video_capture = cv2.VideoCapture(0)

while True:
    # Ler o quadro da câmera
    ret, frame = video_capture.read()

    # Detectar rostos e calcular embeddings na imagem da câmera
    try:
        # Extrair embeddings do rosto detectado
        frame_embedding = DeepFace.represent(img_path=frame, model_name=model_name)[0]["embedding"]

        # Inicializar nome padrão
        name = "Desconhecido"

        # Verificar se o rosto detectado corresponde a alguém conhecido
        for idx, known_embedding in enumerate(known_face_embeddings):
            # Comparar embeddings com DeepFace (menor distância = mais similar)
            result = DeepFace.verify(known_embedding, frame_embedding, model_name=model_name, distance_metric="cosine")
            
            if result["verified"]:  # Verificação positiva
                name = known_face_names[idx]
                break

        # Exibir o nome e caixa ao redor do rosto
        cv2.putText(frame, name, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    except:
        pass

    # Mostrar o quadro com a câmera
    cv2.imshow('Video', frame)

    # Sair do loop ao pressionar 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
video_capture.release()
cv2.destroyAllWindows()
