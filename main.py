import face_recognition
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import streamlit as st

def list_files(folder_path):
    files = []
    for root, dirs, filenames in os.walk(folder_path):
        for filename in filenames:
            files.append(os.path.join(root, filename))
    return files

def face_detection(user_image_path, folder_path):
    user_image = face_recognition.load_image_file(user_image_path)
    gallery_images = [os.path.join(folder_path, img) for img in os.listdir(folder_path)]
    gallery_face_encodings = []
    for img_path in gallery_images:
        img = face_recognition.load_image_file(img_path)
        face_encoding = face_recognition.face_encodings(img)
        if len(face_encoding) > 0:
            gallery_face_encodings.append(face_encoding[0])


    user_face_encoding = face_recognition.face_encodings(user_image)[0]

    matches = face_recognition.compare_faces(gallery_face_encodings, user_face_encoding)
    matched_images = []
    for i in range(0,len(matches)):
        if matches[i] == True:
            matched_images.append(cv2.imread(gallery_images[i]))

    def plot_images(image_list, titles=None, cmap=None):
        if matched_images == []:
            st.error("No faces found in the image.")
        else:
            st.subheader("Images in which the person is found")
            for i in range(len(matched_images)):
                st.image(matched_images[i], use_column_width=True)


    plot_images(matched_images)

def main():
    st.title("Face Recognition App")

    st.header("Step 1: Upload Input Image")
    input_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    st.sidebar.header("Step 2: Select Folder")
    folder_path = st.sidebar.text_input("Enter folder path :")

    if folder_path:
        # Check if the folder path is valid
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            st.sidebar.success("Valid folder path!")

            # List files in the folder
            files = list_files(folder_path)

            # Display the list of files
            st.sidebar.write("### Files in the folder:")
            for file in files:
                st.sidebar.write(file)
        else:
            st.sidebar.error("Invalid folder path. Please enter a valid folder path.")
    
    if input_image is not None:
        st.image(input_image, caption="Uploaded Image.", use_column_width=True)

    if st.button("Search for Faces"):
        if input_image is not None and folder_path:
            face_detection(input_image, folder_path)
        else:
            st.error("Invalid image path.")

if __name__ == "__main__":
    main()  
