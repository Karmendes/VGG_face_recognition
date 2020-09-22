

import cv2
import face_recognition
import numpy as np
from PIL import Image


# Detect faces
def detect_faces(frame,detector):
    """
    Detects faces from frame

    Inputs: frame, detector
    Output: rects
    """
    # convert the input frame from (1) BGR to grayscale (for face
    # detection) and (2) from BGR to RGB (for face recognition)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # detect faces in the grayscale frame
    rects = detector.detectMultiScale(gray, scaleFactor=1.1,
                        minNeighbors=5, minSize=(30, 30),
                        flags=cv2.CASCADE_SCALE_IMAGE)
    return rects

# Get the biggest face
def bigger_face(rects):
    """
    Gets bigger face from all detected faces from the frame

    Input: rects
    Output: rects
    """
    if len(rects) > 0:
        areas = []
        for rect in rects:
            x = rect[0]
            y = rect[1]
            w = rect[2]
            h = rect[3]
            height = y + h
            width = x + w
            area = height* width
            areas.append(area)
        index = areas.index(max(areas))
        return rects[index] 
    else:
        return rects


# Join the functions above
def get_face(frame,detector):
    # detect faces
    rects = detect_faces(frame,detector)    
    # get the biggest face
    rects = bigger_face(rects)
    return rects

# Check if the face has 68 landmarks
def check_landmarks(frame,rects):
    """
    Checks if the face has all landmarks

    Inputs: frame, rects
    Output: Boolean 
    """
    if len(rects) > 0:
        face_image = frame[rects[1]:rects[1]+rects[3],rects[0]:rects[0]+rects[2]]
        ROI = face_recognition.face_landmarks(face_image)
        if len(ROI) > 0:
            return True
        else:
            return False
    else:
        False

# Check if the dimension from face is enough
def check_face_dimension(rects,input_dim_face,result_landmarks):
    """
    Checks if the minimum dimension of the face is greater than the minimum value

    Inputs: rects, input_dim_face
    Output: Boolean
    """
    if result_landmarks:
        face_dim = min(rects[2],rects[3])
        if face_dim > input_dim_face:
            return True
        else:
            return False
    else:
        return False

# Check if the frame is not so blurred
def check_face_blur(rects,frame,input_min_blur,result_face_dimension):
    """
    Checks if the face is too blurred

    Inputs: rects, frame, input_min_blur, check_face_dimension
    Output: Boolean
    """
    if result_face_dimension:
        #get only the image of the face
        face_image = frame[rects[1]:rects[1]+rects[3],rects[0]:rects[0]+rects[2]]
        pil_image = Image.fromarray(face_image)
        open_cv_image = np.array(pil_image)
        gray_face = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
        # check how blury the face is
        face_blur = cv2.Laplacian(gray_face, cv2.CV_64F).var()
        if face_blur > input_min_blur:
            return True
        else:
            return False
    else:
        return False

# Join the checks functions above      
def checks_face(frame,rects,input_dim_face,input_min_blur):
    # check landmarks
    result_landmarks = check_landmarks(frame,rects)
    # check min size head
    result_face_dim = check_face_dimension(rects,input_dim_face,result_landmarks)
    # check blur
    result_blur = check_face_blur(rects,frame,input_min_blur,result_face_dim)
    return result_blur


# Join the get face and checks_face
def pre_process_frame(frame,detector,input_dim_face = 0,input_min_blur = 0):
    rects = get_face(frame,detector)
    check = checks_face(frame,rects,input_dim_face,input_min_blur)
    if check:
        return rects
    else:
        return []

    
    
    
    
    
    
    
    