import os, time, tempfile
import cv2 as cv
from PIL import Image
import streamlit as st
from ultralytics import YOLO



MODEL_DIR = './weight(68)/best.pt'




def main():
    # load a model
    global model
    model = YOLO(MODEL_DIR)

    
    # Define a function to create sidebar headers and lists for each class of animals
    def create_sidebar_header_and_list(classifications):
        
        for category, animal_list in classifications.items():
            st.sidebar.header(f"**{category.capitalize()} Classes**")
            for animal in animal_list:
                st.sidebar.markdown(f"- *{animal.capitalize()}*")

    # Define animal classifications using a dictionary (modify as needed)
    animal_classifications = {
    "Mammals": ["Bear", "Brown bear", "Bull", "Camel", "Cattle", "Cheetah", "Deer", "Elephant", "Fox", "Giraffe", "Goat", "Horse", "Jaguar", "Hippopotamus", "Kangaroo", "Koala", "Lion", "Monkey", "Mouse", "Mule", "Ostrich", "Otter", "Panda", "Pig", "Polar bear", "Rabbit", "Raccoon", "Rhinoceros", "Sheep"],
    "Birds": ["Chicken", "Duck", "Eagle", "Goose", "Owl", "Parrot", "Penguin", "Raven", "Sparrow"],
    "Reptiles": ["Crocodile", "Lizard", "Snake"],
    "Amphibians": ["Frog"],
    "Fish": ["Fish", "Goldfish", "Shark", "Shrimp", "Squid"],
    "Invertebrates": ["Crab", "Jellyfish", "Ladybug", "Magpie", "Moths and butterflies", "Scorpion", "Sea lion", "Sea turtle", "Seahorse", "Snail", "Spider"]
    }

# Call the function with the animal classifications dictionary
    create_sidebar_header_and_list(animal_classifications)
    
    st.title("Animal Species Detection")
    st.write("The aim of this project is to develop an efficient computer vision model capable of real-time wildlife detection.")

    # Load image or video
    uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png', 'mp4'])

    if uploaded_file:
        if uploaded_file.type.startswith('image'):
            inference_images(uploaded_file)
        
        if uploaded_file.type.startswith('video'):
            inference_video(uploaded_file)


def inference_images(uploaded_file):
    image = Image.open(uploaded_file)
     # predict the image
    predict = model.predict(image)

    # plot boxes
    boxes = predict[0].boxes
    plotted = predict[0].plot()[:, :, ::-1]

    if len(boxes) == 0:
        st.markdown("**No Detection**")

    # open the image.
    st.image(plotted, caption="Detected Image", width=600)
    


def inference_video(uploaded_file):
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_file.read())
    temp_file.close()

    cap = cv.VideoCapture(temp_file.name)
    frame_count = 0
    if not cap.isOpened():
        st.error("Error opening video file.")
 

    frame_placeholder = st.empty()
    stop_placeholder = st.button("Stop")

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1
        if frame_count % 2 == 0:
            # predict the frame
            predict = model.predict(frame, conf=0.75)
            # plot boxes
            plotted = predict[0].plot()

            # Display the video
            frame_placeholder.image(plotted, channels="BGR", caption="Video Frame")
        
        # Clean up the temporary file
        if stop_placeholder:
            os.unlink(temp_file.name)
            break

    cap.release()  
    

if __name__=='__main__':
    main()




