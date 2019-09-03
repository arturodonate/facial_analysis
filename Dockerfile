FROM python:3.6-slim-stretch

RUN apt-get update && apt-get install -y \
    build-essential \
    vim \
    cmake \
    curl 

RUN pip install opencv-python \
    scipy \
    torch \
    dlib \
    face_recognition \
    imutils

WORKDIR "/home/facerec"
COPY face_match.py "/home/facerec"
COPY dlib_face_recognition_resnet_model_v1.dat "/home/facerec"
COPY shape_predictor_5_face_landmarks.dat "/home/facerec"
ENTRYPOINT ["/bin/bash"]

