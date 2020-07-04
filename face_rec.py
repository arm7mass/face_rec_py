# face recognition by ARM

# todo: validate the Paths
import face_recognition
import os
import cv2
# from pathlib import Path, PureWindowsPath


KNOWN_FACES_DIR = 'known_faces'
UNKNOWN_FACES_DIR = 'unknown_faces'
TOLERANCE = 0.6
FRAME_THICKNESS = 3
FONT_THICKNESS = 2
MODEL = "hog"  # cnn
# Returns (R, G, B) from name


def name_to_color(name):
    # Take 3 first letters, tolower()
    # lowercased character ord() value rage is 97 to 122, substract 97, multiply by 8
    color = [(ord(c.lower())-97)*8 for c in name[:3]]
    return color


print('loading known faces')

known_faces = []
known_names = []
# file_path_known = Path('D:/Development/Projects/OnGoing_Projects/sentdex_series/face_rec_py/known_faces')
# win_path_known = PureWindowsPath(file_path_known)
# file_path_unknown = Path('D:/Development/Projects/OnGoing_Projects/sentdex_series/face_rec_py/unknown_faces')
# win_path_unknown = PureWindowsPath(file_path_unknown)

# We organize known faces as subfolders of KNOWN_FACES_DIR
# Each subfolder's name becomes our label (name)
for name in os.listdir(KNOWN_FACES_DIR):
    # Next we load every file of faces of known person
    for filename in os.listdir(f"{KNOWN_FACES_DIR}/{name}"):
        # Load an image
        image = face_recognition.load_image_file(f"{KNOWN_FACES_DIR}/{name}/{filename}")
        # Get 128-dimension face encoding
        # Always returns a list of found faces, for this purpose we take first face only
        # (assuming one face per image as you can't be twice on one image)
        encoding = face_recognition.face_encodings(image)[0]
        # Append encodings and name
        known_faces.append(encoding)
        known_names.append(name)

print('processing unknown faces')
# Now let's loop over a folder of faces we want to label
for filename in os.listdir(UNKNOWN_FACES_DIR):
    # Load image
    print(f'filename {filename}', end='')
    image = face_recognition.load_image_file(f"{UNKNOWN_FACES_DIR}/{name}/{filename}")
    # This time we first grab face locations - we'll need them to draw boxes
    locations = face_recognition.face_locations(image, model=MODEL)
    # Now since we know locations, we can pass them to face_encodings as second argument
    # Without that it will search for faces once again slowing down whole process
    encodings = face_recognition.face_encodings(image,  locations)
    # We passed our image through face_locations and face_encodings, so we can modify it
    # First we need to convert it from RGB to BGR as we are going to work with cv2
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # But this time we assume that there might be more faces in an image - we can find faces of
    print(f', found {len(encodings)} face(s)')
    for face_encoding, face_location in zip(encodings, locations):
        # We use compare_faces (but might use face_distance as well)
        # Returns array of True/False values in order of passed known_faces
        results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)
        match = None
        if True in results:
            match = known_names[results.index(True)]
            print(f"Match found: {match} from {results}")
            top_left = (face_location[3], face_location[0])
            bottom_right = (face_location[1], face_location[2])

            color = name_to_color(match)
            # Paint frame
            cv2.rectangle(image, top_left, bottom_right, color, FRAME_THICKNESS)
            # Now we need smaller, filled grame below for a name
            # This time we use bottom in both corners - to start from bottom and move 50 pixels
            top_left = (face_location[3], face_location[2])
            bottom_right = (face_location[1], face_location[2]+22)
            # paint frame
            cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)
            # Wite a name
            cv2.putText(image, match, (face_location[3]+10, face_location[2]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), FONT_THICKNESS)

    # Show image
    cv2.imshow(filename, image)
    cv2.waitkey(10000)

    # cv2.detsroyWindow(filename)


