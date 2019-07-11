import face_recognition
import cv2
import datetime
import time
import pyautogui
import numpy as np
import imutils
from imutils.video import FPS


def write(text,file):
    f = open(file,"a")
    f.write("{}\n".format(text))
    return
logfile=r"face.log"

video_capture = cv2.VideoCapture(0)
# cap.set(CV_CAP_PROP_FPS, 50);
fps = FPS().start()



mani_image = face_recognition.load_image_file("images/mani2.jpeg")
mani_face_encoding = face_recognition.face_encodings(mani_image)[0]


paru_image = face_recognition.load_image_file("images/paru1.jpeg")
paru_face_encoding = face_recognition.face_encodings(paru_image)[0]


ayyapa_image = face_recognition.load_image_file("images/ayyapa.jpg")
ayyapa_face_encoding = face_recognition.face_encodings(ayyapa_image)[0]



# Create arrays of known face encodings and their names
known_face_encodings = [
    mani_face_encoding,
    paru_face_encoding,
    ayyapa_face_encoding
]
known_face_names = [
    "mani",
    "paru",
    "ayyapa"
]

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

currentFrame = 0
while True:

    ret, frame = video_capture.read()
    frame = imutils.resize(frame, width=450)
    
    timer=datetime.datetime.now().strftime("%d %B %y %I:%M %p")
    cv2.putText(frame,timer,(150,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

   
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

   
    rgb_small_frame = small_frame[:, :, ::-1]

    
    if process_this_frame:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        

        face_names = []
       
        for face_encoding in face_encodings:
            
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
                print(name + " " + "is entered into office" +'\t' + timer)

                write(name + " " + "is entered into office" +'\t\t' + timer ,logfile)
                c="/home/pc/Desktop/facepro/known/image_{}.png".format(datetime.datetime.now().strftime("%d %B %y %I:%M %p"))
                cv2.imwrite(c,frame)
                
            else:
                c="/home/pc/Desktop/facepro/unknown/image_{}.png".format(datetime.datetime.now().strftime("%d %B %y %I:%M %p"))
                cv2.imwrite(c,frame)
               

          

            face_names.append(name)

            

    process_this_frame = not process_this_frame


    
    for (top, right, bottom, left), name in zip(face_locations, face_names):
       
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 9, bottom - 6), font, 1.0, (255, 255, 255), 1)

    
    cv2.imshow('Video', frame)

  
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    fps.update()
fps.stop()
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))



video_capture.release()
cv2.destroyAllWindows()