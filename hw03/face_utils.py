cat > face_utils.py << 'EOF'
import face_recognition
import cv2
import numpy as np
import os

def load_known_faces(known_faces_dir="known_faces"):
    """加载已知人脸库"""
    known_encodings = []
    known_names = []
    
    if not os.path.exists(known_faces_dir):
        return known_encodings, known_names
    
    for filename in os.listdir(known_faces_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            name = os.path.splitext(filename)[0]
            image_path = os.path.join(known_faces_dir, filename)
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            
            if encodings:
                known_encodings.append(encodings[0])
                known_names.append(name)
    
    return known_encodings, known_names

def detect_faces(image_array):
    """检测人脸位置"""
    return face_recognition.face_locations(image_array)

def recognize_faces(image_array, known_encodings, known_names, tolerance=0.6):
    """识别人脸"""
    face_encodings = face_recognition.face_encodings(image_array)
    face_locations = face_recognition.face_locations(image_array)
    
    results = []
    for face_encoding, face_location in zip(face_encodings, face_locations):
        name = "未知"
        
        if known_encodings:
            matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance)
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)
            
            if True in matches:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_names[best_match_index]
        
        results.append((face_location, name))
    
    return results

def draw_results(image_array, results):
    """绘制结果"""
    if len(image_array.shape) == 3 and image_array.shape[2] == 3:
        img = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    else:
        img = image_array.copy()
    
    for (top, right, bottom, left), name in results:
        cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
        
        label = name
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(img, (left, top - label_size[1] - 5), 
                     (left + label_size[0] + 5, top), (0, 255, 0), -1)
        cv2.putText(img, label, (left + 2, top - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
EOF
