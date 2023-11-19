import openai
import math
import cv2
import dlib
import numpy as np
from matplotlib import pyplot as plt
import streamlit as st
from deepface import DeepFace
from openai.error import AuthenticationError

st.set_page_config(initial_sidebar_state='collapsed')
st.set_option('deprecation.showPyplotGlobalUse', False)

st.title("Analyzing Emotions - :mag: :smiling_face_with_tear:")
st.text(
    "A bot which analyzes your mood (using an image) and provides feedback/motivation\nbased on the emotion depicted from the provided image! \n(IMPORTANT: the bot is NOT always accurate in detecting the right emotions)")

st.subheader(":star2: Available features: ")
MotivateGPT = st.checkbox('Enable MotivationGPT?', value=True)
st.divider()

uploaded_file = st.file_uploader("Upload your file here...", type=["jpg", "jpeg", "png"])

# Load's dlib's pretrained face detector model
frontalface_detector = dlib.get_frontal_face_detector()
#Load the 68 face Landmark file
landmark_predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')

# Converts dlib rectangular object to box coordinates
def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return (x, y, w, h)

#Detects the face in the given image
def detect_face(img):
    try:
        # Detect faces using dlib model
        rects = frontalface_detector(img, 1)

        if len(rects) < 1:
            st.write("No Face Detected")
            return

        # Loop over the face detections
        for (i, rect) in enumerate(rects):
            # Converts dlib rectangular object to bounding box coordinates
            (x, y, w, h) = (rect.left(), rect.top(), rect.width(), rect.height())
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the image with detected faces using Streamlit
        st.image(img, channels="BGR")
        return img

    except Exception as e:
        st.write("Error detecting faces:", e)

def get_landmarks(img):
  #Detect the Faces within the image
  faces = frontalface_detector(img, 1)
  if len(faces):
    landmarks = [(p.x, p.y) for p in landmark_predictor(img, faces[0]).parts()]
  else:
    return None,None

  return img,landmarks

def plot_image_landmarks(image, face_landmarks):
    radius = -1
    circle_thickness = 5
    image_copy = image.copy()
    for (x, y) in face_landmarks:
        cv2.circle(image_copy, (x, y), circle_thickness, (255, 0, 0), radius)

    st.image(image_copy, channels="BGR")  # Display the image in Streamlit
def euclidean_distance(p1,p2):
  """
  type p1, p2 : tuple
  rtype distance: float
  """
  ### YOUR CODE HERE
  dist = math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
  # print(dist)
  return dist
  ### END CODE

def get_pixels_image(img_pixels, plt_flag):
    width = 48
    height = 48

    image = np.fromstring(img_pixels, dtype=np.uint8, sep=" ").reshape((height, width))

    if plt_flag:
        plt.imshow(image, interpolation='nearest', cmap="Greys_r")
        plt.xticks([]); plt.yticks([])

        # Convert the Matplotlib plot to a Streamlit-compatible format
        st.pyplot()
    return image


def detect_face(img):
    try:
        # Your existing face detection code
        # ...

        return img

    except Exception as e:
        st.write("Error detecting faces:", e)

if(MotivateGPT):
    openai.api_key = st.text_input(
        label="For the generation process to work, since this bot uses gpt3, it requires an Open API Key to continue.",
        placeholder="Your key here...", type="password")

def askGPT(text):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": text}
        ],
        max_tokens=400,
        temperature=0.6
    )
    return response.choices[0].message.content

if uploaded_file is not None:
    # Read and process the uploaded image
    image_data = uploaded_file.read()
    nparr = np.fromstring(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Detect landmarks
    with st.spinner("Loading..."):
        img = detect_face(image)

        # Analyze emotions using DeepFace
        emotions = DeepFace.analyze(img, actions=['emotion'])
        dominant_emotion = emotions[0]['dominant_emotion']
        target_emotion = dominant_emotion  # Replace with the emotion you're interested in

        # Initialize variables to store the maximum emotion and its probability
        max_emotion_prob = 0.0

        # Iterate through the detected faces and find the maximum emotion
        for face in emotions:
            emotion_prob = face['emotion'][target_emotion]
            if emotion_prob > max_emotion_prob:
                max_emotion_prob = emotion_prob

        st.subheader(f"{dominant_emotion}: {max_emotion_prob}%")
        st.image(img, channels="BGR")

        st.header("Facial landmarks: ")
        img, landmarks = get_landmarks(image)


    if landmarks is not None:
        # Display the image with landmarks in Streamlit
        plot_image_landmarks(img, landmarks)
    else:
        st.write("No face detected in the uploaded image.")

    if (openai.api_key and MotivateGPT):
        prompt = f"""
                Hey, I am feeling {dominant_emotion}. based on that could you give me a proper feedback\n
                on how I could be better (if i am feeling unwell) or acknowledge the fact that I am in a\n
                good mood? Act as a therapist and try to find a efficient feedbacl based on my emotion!\n
                Try and avoid any inappopriate responses. Please provide a brief response in 50 words or less.
                        """
        try:
            st.divider()
            st.header("MotivateGPT: ")
            with st.spinner("Generating..."):
                generated_text = askGPT(prompt)
            st.success('Your draft is ready!')
            st.write(generated_text)
        except AuthenticationError as e:
            st.error("Please provide a valid OpenAI Api key!")

