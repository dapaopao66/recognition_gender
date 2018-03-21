
import os
import numpy as np
import cv2


# 循环读取图片，添加到一个列表，每个图片为128 *128
def get_file(path):
    IMAGE_SIZE = 128
    images_list = []
    labels_list = []
    counter = 0
    for child_dir in os.listdir(path):
        child_path = os.path.join(path, child_dir)
        # print(child_path)
        for dir_image in os.listdir(child_path):
            # print(dir_image)
            if dir_image.endswith('.jpg'):
                img = cv2.imread(os.path.join(child_path, dir_image))
                resized_img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
                colored_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY) #转为灰度图
                images_list.append(colored_img)
                labels_list.append(counter)

        counter += 1

    images_list = np.array(images_list)

    return images_list,labels_list,counter

def get_file_name(path):
    name_list = []
    for child_dir in os.listdir(path):
        name_list.append(child_dir)
    return name_list


if __name__ == '__main__':
    # img_list, label_lsit, counter = get_file('images')
    # print(counter)
    # print(label_lsit)
    # print(len(img_list))
    print(get_file_name('images'))