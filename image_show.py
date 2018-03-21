import cv2
import os
from image_train import Model
from handle_image import get_file_name

if __name__ == '__main__':
    IMAGE_SIZE = 128
    name_list = get_file_name('gender_image')
    print(name_list)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    model = Model()
    model.load()
    frame = cv2.imread('1.jpg')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    print(len(faces))
    for (x, y, w, h) in faces:
        ROI = gray[x:x + w, y:y + h]
        ROI = cv2.resize(ROI, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
        label, prob = model.predict(ROI)
        print(prob)
        if prob > 0.5:
            show_name = name_list[label]
            if (show_name == '0'):
                show_name = "female"
            else:
                show_name = "male"
            print(name_list[label])
        else:
            show_name = 'unknow'
        cv2.putText(frame, show_name, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)  # 显示名字
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # 在人脸区域画一个正方形出来
    cv2.imshow("frame", frame)
    cv2.waitKey(0)

    cv2.destroyAllWindows()