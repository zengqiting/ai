cat > app.py << 'EOF'
import streamlit as st
import face_recognition
from PIL import Image
import numpy as np
import os
from face_utils import load_known_faces, detect_faces, recognize_faces, draw_results

# 页面配置
st.set_page_config(
    page_title="人脸识别系统",
    page_icon="👤",
    layout="wide"
)

# 标题
st.title("👤 人脸识别系统")
st.markdown("基于 `face_recognition` 和 `Streamlit` 构建的人脸检测与识别应用")

# 侧边栏配置
with st.sidebar:
    st.header("⚙️ 配置")
    
    mode = st.radio(
        "选择模式",
        ["人脸检测", "人脸识别"],
        help="人脸检测：只检测人脸位置；人脸识别：与已知人脸库比对"
    )
    
    if mode == "人脸识别":
        tolerance = st.slider(
            "识别阈值", 
            min_value=0.4, 
            max_value=0.8, 
            value=0.6, 
            step=0.01,
            help="数值越小，匹配越严格"
        )
    else:
        tolerance = 0.6
    
    st.divider()
    st.subheader("📁 已知人脸库")
    
    known_faces_dir = "known_faces"
    if os.path.exists(known_faces_dir):
        faces = [f for f in os.listdir(known_faces_dir) 
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if faces:
            st.success(f"已加载 {len(faces)} 个人脸样本")
        else:
            st.warning("人脸库为空")
    else:
        st.info("未找到 known_faces 目录，仅支持检测模式")

# 主区域
col1, col2 = st.columns(2)

with col1:
    st.subheader("📤 上传图片")
    
    uploaded_file = st.file_uploader(
        "选择图片",
        type=['jpg', 'jpeg', 'png']
    )
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="原始图片", use_container_width=True)

with col2:
    st.subheader("✨ 处理结果")
    
    if uploaded_file:
        with st.spinner("处理中..."):
            image_array = np.array(image)
            
            if mode == "人脸识别":
                known_encodings, known_names = load_known_faces(known_faces_dir)
                if not known_encodings:
                    st.warning("人脸库为空，仅执行人脸检测")
                    results = [(loc, "人脸") for loc in detect_faces(image_array)]
                else:
                    results = recognize_faces(image_array, known_encodings, known_names, tolerance)
            else:
                face_locations = detect_faces(image_array)
                results = [(loc, "人脸") for loc in face_locations]
            
            if results:
                result_image = draw_results(image_array, results)
                st.image(result_image, caption=f"检测到 {len(results)} 张人脸", use_container_width=True)
                
                with st.expander("📋 详细信息"):
                    for i, (location, name) in enumerate(results, 1):
                        top, right, bottom, left = location
                        st.write(f"**人脸 {i}**: {name}")
                        st.write(f"  位置: (上:{top}, 右:{right}, 下:{bottom}, 左:{left})")
            else:
                st.warning("未检测到人脸")
                st.image(image_array, caption="未检测到人脸", use_container_width=True)
    else:
        st.info("👈 请先上传图片")

st.divider()
st.markdown("""
### 📝 使用说明
1. **人脸检测模式**：上传图片后自动检测所有人脸位置
2. **人脸识别模式**：需要先在 `known_faces` 目录放入已知人脸的图片
3. 识别阈值：数值越小匹配越严格，推荐 0.6
""")
EOF
