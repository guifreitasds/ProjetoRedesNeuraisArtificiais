import cv2
from deepface import DeepFace

# Lista de imagens conhecidas e nomes associados
known_face_images = ["./imagesdb/eugui.jpeg", "./imagesdb/dido.jpeg", "./imagesdb/dido2.jpeg"]
known_face_names = ["Guilherme Freitas", "Danilo Pereira", "Danilo Pereira"]
#model_name = "VGG-Face"
#model_name = "OpenFace"
#model_name = "DeepID"
model_name = "Facenet512"

last_recognized_name = "Unknown"
recognition_timeout = 100  # Número de frames para reter o rosto reconhecido
recognition_counter = 0


# Carregar imagens e calcular embeddings 
known_face_embeddings = []
for image_path in known_face_images:
    embedding = DeepFace.represent(img_path=image_path, model_name=model_name)[0]["embedding"]
    known_face_embeddings.append(embedding)


video_capture = cv2.VideoCapture(0)

while True:

    ret, frame = video_capture.read()

    if recognition_counter == 0:
        try:
            frame_embedding = DeepFace.represent(img_path=frame, model_name=model_name)[0]["embedding"]

            name = "Desconhecido"
            for idx, known_embedding in enumerate(known_face_embeddings):
                result = DeepFace.verify(known_embedding, frame_embedding, model_name=model_name, distance_metric="cosine")
                
                if result["verified"]:
                    name = known_face_names[idx]
                    last_recognized_name = name  # Armazena o nome do rosto reconhecido
                    recognition_counter = recognition_timeout  # Reseta o contador
                    break
            
            # Exibir o nome e caixa ao redor do rosto
            cv2.putText(frame, name, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        except Exception as e:
            pass

    else:
        # Exibir o último rosto reconhecido sem fazer um novo reconhecimento
        cv2.putText(frame, last_recognized_name, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        recognition_counter -= 1  # Decrementa o contador


    cv2.imshow('Video', frame)

    # Sair do loop ao pressionar 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
