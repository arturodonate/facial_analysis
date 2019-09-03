import sys
import dlib
from scipy.spatial import distance

# Load model parameters from files
detector = dlib.get_frontal_face_detector()
shape_pred = dlib.shape_predictor("shape_predictor_5_face_landmarks.dat")
facerec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# Detect a single face from an image
def detect_face_from_image(image):
    faces = detector(image, 1)
    if len(faces) > 1:
        print("WARNING: Detected more than 1 face in an image. Defaulting to largest bounding box.")
        # TODO: NEED TO HANDLE CASE WHERE >1 FACE IS FOUND
    return faces[0]

# Detect face landmarks and build the descriptor
def build_face_descriptor(face, image):
    face_shape = shape_pred(image, face)
    face_descriptor = facerec.compute_face_descriptor(image,face_shape)
    return face_descriptor

# Calculate similarity between 2 faces
def face_verification(descriptor1, descriptor2, threshold):
    descriptor_distance = distance.euclidean(descriptor1, descriptor2)
    if descriptor_distance < distance_threshold:
        return 1 # Match
    else:
        return 0 # No match

if __name__ == "__main__":
    # Check arguments
    if len(sys.argv) < 3:
        print("USAGE: \"python face_match.py [PATH_TO_IMAGE1] [PATH_TO_IMAGE2]\"")
        sys.exit(0)
    
    # Set distance thredhold
    distance_threshold = 0.6
    # Load image pair
    img1 = dlib.load_rgb_image(sys.argv[1])
    img2 = dlib.load_rgb_image(sys.argv[2])
    # Extract faces
    faces1 = detect_face_from_image(img1)
    faces2 = detect_face_from_image(img2)
    # Build descriptors
    face_descriptor1 = build_face_descriptor(faces1, img1)
    face_descriptor2 = build_face_descriptor(faces2, img2)
    # Determine of the 2 faces match
    verification_result = face_verification(face_descriptor1, face_descriptor2, distance_threshold) 
    if verification_result == 1:
        print("MATCH")
    else:
        print("NO MATCH")

