import cv2

# Coba backend lain seperti CAP_DSHOW atau CAP_MSMF
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # atau cv2.CAP_MSMF

if not cap.isOpened():
    print("Tidak bisa membuka kamera")
    exit()

# Mengatur resolusi kamera
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Lebar frame
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Tinggi frame

while True:
    ret, frame = cap.read()
    if not ret:
        print("Tidak bisa menangkap frame")
        break

    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
