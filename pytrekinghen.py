import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

print("Sedang membaca data dan melatih AI...")
# 1. BACA DATA & LATIH AI (KNN)
try:
    df = pd.read_csv('dataset_gestur.csv')
except FileNotFoundError:
    print("Error: File dataset_gestur.csv tidak ditemukan!")
    exit()

X = df.drop('label', axis=1).values  # Koordinat 21 titik
y = df['label'].values               # Label gestur (0, 1, 2, 3)

# Latih model K-Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X, y)
print("AI berhasil dilatih dan siap digunakan!")

# Daftar nama dan warna neon untuk tiap gestur
label_map = {
    0: "GESTUR: MENGEPAL",
    1: "GESTUR: THUMBS UP",
    2: "GESTUR: LOVE / SARANGHAE",
    3: "GESTUR: SAY HI"
}
color_map = {
    0: (255, 255, 255), # Putih
    1: (0, 255, 0),    # Hijau
    2: (0, 0, 255),    # Merah
    3: (255, 0, 255)   # Ungu
}

# 2. INISIALISASI KAMERA & MEDIAPIPE
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    img = cv2.resize(img, (854, 480))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    
    h, w, c = img.shape
    neon_canvas = np.zeros((h, w, 3), np.uint8)

    gesture_text = "Mencari Gestur..."
    neon_color = (255, 255, 0) 

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            
            base_x = hand_landmarks.landmark[0].x
            base_y = hand_landmarks.landmark[0].y
            base_z = hand_landmarks.landmark[0].z
            
            landmark_list = []
            for lm in hand_landmarks.landmark:
                landmark_list.extend([lm.x - base_x, lm.y - base_y, lm.z - base_z])
                
            prediksi = knn.predict([landmark_list])[0]
            
            gesture_text = label_map.get(prediksi, "TIDAK DIKENAL")
            neon_color = color_map.get(prediksi, (255, 255, 0))

            mp_draw.draw_landmarks(
                neon_canvas, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS,
                connection_drawing_spec=mp_draw.DrawingSpec(color=neon_color, thickness=2, circle_radius=1),
                landmark_drawing_spec=mp_draw.DrawingSpec(color=(0, 255, 255), thickness=1, circle_radius=2)
            )

        neon_blur = cv2.GaussianBlur(neon_canvas, (11, 11), 0)
        combined = cv2.addWeighted(img, 1.0, neon_canvas, 1.0, 0)
        final_display = cv2.addWeighted(combined, 1.0, neon_blur, 0.5, 0)
    else:
        final_display = img
        gesture_text = "Tampilkan Tangan"

    cv2.putText(final_display, gesture_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow("AI Hand Tracking", final_display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()