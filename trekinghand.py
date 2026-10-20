import cv2
import mediapipe as mp
import csv
import os

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Kita batasi 1 tangan dulu agar datanya fokus dan tidak tertukar
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1, 
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

# Nama file penyimpan data
csv_file = 'dataset_gestur.csv'

# Buat header CSV jika file belum ada (Label + 21 titik x, y, z)
if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        header = ['label']
        for i in range(21):
            header.extend([f'x{i}', f'y{i}', f'z{i}'])
        writer.writerow(header)

print("=== PEREKAM DATASET AKTIF ===")
print("Tekan dan TAHAN tombol angka di layar aktif untuk merekam:")
print("[0] = Mengepal (Fist)")
print("[1] = Thumbs Up")
print("[2] = Love / Saranghae")
print("[3] = Say Hi / Melambai")
print("Tekan [Q] untuk keluar.")

with open(csv_file, mode='a', newline='') as f:
    writer = csv.writer(f)
    
    while True:
        success, img = cap.read()
        if not success:
            break

        img = cv2.flip(img, 1)
        img = cv2.resize(img, (854, 480))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        
        key = cv2.waitKey(1) & 0xFF
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                base_x = hand_landmarks.landmark[0].x
                base_y = hand_landmarks.landmark[0].y
                base_z = hand_landmarks.landmark[0].z
                
                landmark_list = []
                for lm in hand_landmarks.landmark:
                    landmark_list.extend([lm.x - base_x, lm.y - base_y, lm.z - base_z])
                
                if key == ord('0'):
                    writer.writerow([0] + landmark_list)
                    print("Terekam -> Gestur: [0] Mengepal")
                elif key == ord('1'):
                    writer.writerow([1] + landmark_list)
                    print("Terekam -> Gestur: [1] Thumbs Up")
                elif key == ord('2'):
                    writer.writerow([2] + landmark_list)
                    print("Terekam -> Gestur: [2] Love")
                elif key == ord('3'):
                    writer.writerow([3] + landmark_list)
                    print("Terekam -> Gestur: [3] Say Hi")

        cv2.putText(img, "Tahan angka 0/1/2/3 untuk rekam. Q untuk keluar.", (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.imshow("Perekam Dataset ML", img)

        if key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()