import random
import numpy as np
from handle_image import get_file
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.models import Sequential,load_model
from keras.layers import Dense,Activation,Convolution2D,MaxPooling2D,Flatten,Dropout
from keras.optimizers import Adam

#建立数据
class DataSet(object):
    def __init__(self):
        self.nb_classes = None
        self.X_train = None
        self.X_test = None
        self.Y_train = None
        self.Y_test = None
        self.img_size = 128

    def extract_data(self,train_path):
        imgs, labels, counter = get_file(train_path)
        print(labels)
        # 避免过拟合，采用交叉验证，验证集占训练集30%，固定随机种子（random_state)
        X_train, X_test, y_train, y_test = train_test_split(imgs, labels, test_size=0.3,
                                                            random_state=random.randint(0, 100))

        #数据预处理 keras backend 用的TensorFlow 黑白图片 channel 1
        X_train = X_train.reshape(X_train.shape[0], 1, self.img_size, self.img_size) / 255.
        X_test = X_test.reshape(X_test.shape[0], 1, self.img_size, self.img_size) / 255.

        #label 转为 one-hot 数据
        Y_train = np_utils.to_categorical(y_train, num_classes=counter)
        Y_test = np_utils.to_categorical(y_test, num_classes=counter)

        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test
        self.nb_classes = counter


#建立model  使用CNN（卷积神经网络）
class Model(object):
    FILE_PATH = "store/model.h5"
    IMAGE_SIZE = 128
    def __init__(self):
        self.model = None

    def build_model(self,dataset):
        self.model = Sequential()
        #进行一层卷积 输出 shape (32,128,128)
        self.model.add(Convolution2D(filters=32,kernel_size=5,strides=1, padding='same',data_format='channels_first', input_shape=dataset.X_train.shape[1:]))
        #使用relu激励函数
        self.model.add(Activation('relu'))
        #池化，输出为shape (32,64,64)
        self.model.add(MaxPooling2D(pool_size=2,strides=2,padding='same',data_format='channels_first'))
        #dropout 防止过拟合
        self.model.add(Dropout(0.25))

        #进行一层卷积 输出为shape (64,32,32)
        self.model.add(Convolution2D(64, 5, strides=1, padding='same', data_format='channels_first'))
        # 使用relu激励函数
        self.model.add(Activation('relu'))
        # 池化，输出为原来的一半 shape (64,32,32)
        self.model.add(MaxPooling2D(2, 2, 'same', data_format='channels_first'))
        # dropout 防止过拟合
        self.model.add(Dropout(0.25))

        #全连接层
        self.model.add(Flatten())
        self.model.add(Dense(512))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(dataset.nb_classes))
        self.model.add(Activation('softmax'))

        self.model.summary()

    def train(self,dataset):
        adam = Adam(lr=1e-4)
        self.model.compile(optimizer=adam,
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        # epochs 循环次数  batch_size 批处理大小
        self.model.fit(dataset.X_train, dataset.Y_train, epochs=25, batch_size=32, )

    def save(self, file_path=FILE_PATH):
        print('Model 保存.')
        self.model.save(file_path)

    def load(self, file_path=FILE_PATH):
        print('Model 读取.')
        self.model = load_model(file_path)

    #预测
    def predict(self,img):
        img = img.reshape((1, 1, self.IMAGE_SIZE, self.IMAGE_SIZE))
        img = img.astype('float32')
        img = img/255.0

        result = self.model.predict_proba(img)  #预测图像结果
        max_index = np.argmax(result)   #取平局值最大
        print("begin")
        print(result)
        print(max_index)
        print(result[0][max_index])
        print("end")
        return max_index,result[0][max_index]  #第一个参数为概率最高的label的index,第二个参数为对应概率

    def evaluate(self, dataset):
        loss,score = self.model.evaluate(dataset.X_test, dataset.Y_test, verbose=0)
        # print("%s: %.2f%%" % (self.model.metrics_names[1], score[1] * 100))
        print('\ntest loss: ', loss)
        print('\ntest accuracy: ', score)

if __name__ == '__main__':
    dataset = DataSet()
    dataset.extract_data('gender_image')

    model = Model()
    model.build_model(dataset)
    model.train(dataset)
    model.save()

    model = Model()
    model.load()
    model.evaluate(dataset)
