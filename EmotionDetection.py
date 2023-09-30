import cv2
import dlib
import pickle
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import urllib.request
from sklearn import metrics
from scipy.spatial import distance
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import io

import streamlit as st

st.set_page_config(initial_sidebar_state='collapsed')
st.set_option('deprecation.showPyplotGlobalUse', False)


st.title("GrantGPT - :book: :chart_with_upwards_trend:")
st.text(
    "A bot which allows you to generate a grant, as well as analyze how good \nyour own writing is based on the given "
    "context!")

st.subheader(":star2: Available features: ")
analyze = st.checkbox('Analyze Grants', value=True)
generate = st.checkbox('Generate Grants')
st.divider()

frontalface_detector = dlib.get_frontal_face_detector()

''' Converts dlib rectangular object to box coordinates '''
def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return (x, y, w, h)

"""Detects the face in the given image"""
def detect_face(image_url):
  """
  :type image_url: str
  :rtype: None

  """
  try:

    #Decodes image address to cv2 object
    url_response = urllib.request.urlopen(image_url)
    img_array = np.array(bytearray(url_response.read()), dtype=np.uint8)
    image = cv2.imdecode(img_array, -1)

  except Exception as e:
    return "Please check the URL and try again!"

  #Detect faces using dlib model
  rects = frontalface_detector(image, 1)

  if len(rects) < 1:
    return "No Face Detected"

  # Loop over the face detections
  for (i, rect) in enumerate(rects):
    # Converts dlib rectangular object to bounding box coordinates
    (x, y, w, h) = rect_to_bb(rect)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
  plt.imshow(image, interpolation='nearest')
  plt.axis('off')
  plt.show()

detect_face(input('Enter the URL of the image: '));