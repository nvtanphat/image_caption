import streamlit as st
import numpy as np
from keras.preprocessing.sequence import pad_sequences 
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import pickle


#Hamm sinh caption tu anh va hien thi
def generate_and_display_caption(image_path, model_path, tokenizer_path, feature_extractor_path, max_length=34,img_size=224):
    # Load model captionali(cnn+lstm) va fearture extrractor
    caption_model = load_model(model_path)
    feature_extractor = load_model(feature_extractor_path)
    # Load tokenizer từ file pickle
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)

    # Tiền xử lý ảnh
    img = load_img(image_path, target_size=(img_size, img_size))
    img = img_to_array(img) / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)
    # Trích đặc trưng ảnh bằng feature extractor

    image_features = feature_extractor.predict(img, verbose=0)  # Extract image features

    # Sinh caption dần dần bằng greedy search
    in_text = "startseq"
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        #Model du doan tu tiep theo
        yhat = caption_model.predict([image_features, sequence], verbose=0)
        # Lấy index từ có xác suất cao nhất
        yhat_index = np.argmax(yhat)
        word = tokenizer.index_word.get(yhat_index, None)
        if word is None:
            break
        in_text += " " + word
        if word == "endseq":
            break
    caption = in_text.replace("startseq", "").replace("endseq", "").strip()
    # Hiển thị ảnh và caption bằng matplotlib + Streamlit
    img = load_img(image_path, target_size=(img_size, img_size))
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.axis('off')
    plt.title(caption, fontsize=16, color='blue')
    st.pyplot(plt)   # render lên giao diện Streamlit
    st.pyplot(plt.gcf())
    plt.clf()


# Giao diện Streamlit
def main():
    st.title("Trình tạo chú thích hình ảnh")
    st.write("Tải lên hình ảnh và tạo chú thích bằng mô hình đã được đào tạo.")

    # Cho phép upload ảnh
    uploaded_image = st.file_uploader("Chọn một hình ảnh...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Lưu ảnh upload tạm
        with open("uploaded_image.jpg", "wb") as f:
            f.write(uploaded_image.getbuffer())


        model_path = "models/model.keras"
        tokenizer_path = "models/tokenizer.pkl"
        feature_extractor_path = "models/feature_extractor.keras"

        # Sinh caption và hiển thị
        generate_and_display_caption("uploaded_image.jpg", model_path, tokenizer_path, feature_extractor_path)


if __name__ == "__main__":
    main()
