from ultralytics import YOLO
import cv2
from flask import Flask, Response, jsonify

# Global variable to store detected nutritional information
detected_nutrition_info = {}

def video_detection(path_x):
    video_capture = path_x
    # Create a Webcam Object
    cap = cv2.VideoCapture(video_capture)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    model = YOLO("../Model/best.pt")
    classNames = ["Umbi bengkuang", "Umbi kentang", "Umbi kuning", "Umbi singkong", "Umbi talas", "Umbi ungu", "Umbi wortel"]
    
    # Define nutritional information and recommendations for each class
    class_nutrition = {
        "Umbi bengkuang": {
            "Karbohidrat": "9 gram",
            "Gula": "1,8 gram",
            "Rekomendasi": "Konsumsi bengkuang dalam bentuk segar atau dijadikan salad untuk mendapatkan manfaat serat dan nutrisi tanpa meningkatkan kadar gula darah terlalu tinggi."
        },
        "Umbi kentang": {
            "Karbohidrat": "17 gram",
            "Gula": "0,8 gram",
            "Rekomendasi": "Sebaiknya kentang dikonsumsi dengan cara dipanggang atau direbus, bukan digoreng, untuk menghindari tambahan lemak dan kalori. Kombinasikan dengan sayuran lain untuk menu seimbang."
        },
        "Umbi kuning": {
            "Karbohidrat": "27 gram",
            "Gula": "0,5 gram",
            "Rekomendasi": "Konsumsi dengan cara direbus atau dipanggang, dan hindari penambahan gula atau bahan pemanis lainnya. Cocok sebagai sumber energi dalam makanan utama."
        },
        "Umbi singkong": {
            "Karbohidrat": "38,1 gram",
            "Gula": "1,7 gram",
            "Rekomendasi": "Sebaiknya dikonsumsi dalam porsi kecil untuk menghindari peningkatan kadar gula darah yang signifikan. Olah dengan cara direbus atau dikukus, dan hindari konsumsi dalam bentuk gorengan atau yang diberi tambahan gula."
        },
        "Umbi talas": {
            "Karbohidrat": "26 gram",
            "Gula": "0,5 gram",
            "Rekomendasi": "Sebaiknya dikonsumsi dalam porsi sedang dan diolah dengan cara direbus atau dikukus untuk menjaga kandungan nutrisinya."
        },
        "Umbi ungu": {
            "Karbohidrat": "20 gram",
            "Gula": "4,2 gram",
            "Rekomendasi": "Konsumsi dalam porsi yang terkontrol, dan olah dengan cara yang sehat seperti direbus atau dipanggang. Kandungan antioksidan yang tinggi pada umbi ungu juga memberikan manfaat tambahan untuk kesehatan."
        },
        "Umbi wortel": {
            "Karbohidrat": "10 gram",
            "Gula": "4,7 gram",
            "Rekomendasi": "Sebaiknya dikonsumsi dalam porsi kecil, terutama jika Anda perlu mengontrol asupan gula. Wortel dapat dimakan mentah sebagai camilan atau dimasak dalam sup dan tumisan."
        }
    }
    
    # Define a dictionary to map class names to colors
    class_colors = {
        "Umbi bengkuang": (255, 0, 0),    # Red
        "Umbi kentang": (0, 255, 0),      # Green
        "Umbi kuning": (0, 0, 255),       # Blue
        "Umbi singkong": (255, 255, 0),   # Cyan
        "Umbi talas": (255, 0, 255),      # Magenta
        "Umbi ungu": (0, 255, 255),       # Yellow
        "Umbi wortel": (128, 0, 128)      # Purple
    }

    while True:
        success, img = cap.read()
        if not success:
            break

        results = model(img, stream=True)
        detected_nutrition_info = {}

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                cls = int(box.cls[0])
                class_name = classNames[cls]
                color = class_colors[class_name]
                
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                
                nutrition_info = class_nutrition[class_name]
                detected_nutrition_info[class_name] = nutrition_info
                
                label = f'{class_name}'
                t_size = cv2.getTextSize(label, 0, fontScale=0.5, thickness=1)[0]
                c2 = x1 + t_size[0], y1 - t_size[1] - 3
                cv2.rectangle(img, (x1, y1), c2, color, -1, cv2.LINE_AA)
                y0, dy = y1 - t_size[1] - 5, 20
                for i, line in enumerate(label.split('\n')):
                    y = y0 + i * dy
                    cv2.putText(img, line, (x1, y), 0, 0.5, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
        
        yield img, detected_nutrition_info

    cap.release()
    cv2.destroyAllWindows()
