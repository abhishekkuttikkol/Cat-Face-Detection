import cv2

trained_face_data = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalcatface.xml')
cam = cv2.VideoCapture('cat.mp4')
# cam.open('https://192.168.1.199:8080/video')
while cam.isOpened:
    succesful_frame_read,img = cam.read()
    grey_scale = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    face_coordinates = trained_face_data.detectMultiScale(grey_scale)
    print(face_coordinates)

    # a,b,c,d = face_coordinates[0]
    for a,b,c,d in face_coordinates:
        cv2.rectangle(img, (a,b),(c+a,d+b),(0,255,0),2)

        cv2.putText(img, "Cat", (a, b - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow("face Detection",img)
    key = cv2.waitKey(1)
     
    if key == 81:
        break
cv2.destroyAllWindows()