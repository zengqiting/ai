import streamlit as st
import face_recognition
import cv2
import numpy as np
from PIL import Image

st.title("人脸识别系统 (HW03)")
st.write("上传图片检测人脸")

uploaded_file = st.file_uploader("选择图片", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="上传的图片", use_column_width=True)
    
    # 转换为numpy数组
    img_array = np.array(image)
    
    # 人脸检测
    face_locations = face_recognition.face_locations(img_array)
    
    # 在图片上画框
    for top, right, bottom, left in face_locations:
        cv2.rectangle(img_array, (left, top), (right, bottom), (0, 255, 0), 2)
    
    # 显示结果
    st.image(img_array, caption=f"检测到 {len(face_locations)} 张人脸", use_column_width=True)
    
    # 提取特征（可选）
    if face_locations:
        face_encodings = face_recognition.face_encodings(img_array, face_locations)
        st.write(f"提取到 {len(face_encodings)} 个128维特征向量")
