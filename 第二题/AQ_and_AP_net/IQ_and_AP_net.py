import torch
import tensorflow as tf
import os,json
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["THEANO_FLAGS"] = "device=cuda,floatX=float32"
import numpy as np
import tensorflow.keras.models as models
from tensorflow.keras.layers import Reshape,Dense,Dropout,Activation,Flatten,LSTM
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, BatchNormalization
import matplotlib.pyplot as plt
import pickle as cPickle
from scipy.stats import moment

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
# Load the dataset ...
#  You will need to seperately download or generate this file
dataFile = "././modu_signal/RML2016.10a_dict.pkl"
with open(dataFile, 'rb') as f:
    Xd = cPickle.load(f, encoding='latin1')
snrs,mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1,0])
# print('torch GPU:', torch.cuda.is_available())
# print('tensorflow GPU:', tf.test.is_gpu_available())
#   以字典形式存储数据
X = []
lbl = []
for mod in mods:
    for snr in snrs:
        X.append(Xd[(mod,snr)])
        for i in range(Xd[(mod,snr)].shape[0]):  lbl.append((mod,snr))
X = np.vstack(X)
# Partition the data
#  into training and test sets of the form we can train/test on
#  while keeping SNR and Mod labels handy for each
#   设置随机种子（确保每次运行代码选择的数据划分都是一样的）
np.random.seed(2016)
#   定义数据集大小
n_examples = X.shape[0]
#   取一半的数据集作为训练集
n_train = n_examples * 0.5
#   选择训练和测试索引
train_idx = np.random.choice(range(0,n_examples), size=int(n_train), replace=False)
test_idx = list(set(range(0,n_examples))-set(train_idx))
X_train = X[train_idx]
X_test = X[test_idx]


#   将标签转换为one-hot编码形式
def to_onehot(yy):
    yy1 = np.zeros([len(yy), max(yy)+1])
    yy1[np.arange(len(yy)),yy] = 1
    return yy1


Y_train = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), train_idx)))
Y_test = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), test_idx)))
#   从训练数据中提取应该输入的形状并打印出来
in_shp = list(X_train.shape[1:])
print(X_train.shape, in_shp)
classes = mods


#Z数据 人工特征信息提取（4阶累积量）
# Z = []
# for n in range(n_examples):
#     Z[n] = moment(X[n],moment=4) - 3 * (moment(X[n], moment=2))**2
# z_examples = Z.shape[0]
# z_train = int(z_examples *0.5)
# train_idz = np.random.choice(range(0, z_examples), size=z_train, replace=False)
# test_idz = list(set(range(0,z_examples))-set(train_idz))
# Z_train = Z[train_idz]
# Z_test = Z[test_idz]
#
# ZZ_train = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), train_idz)))
# ZZ_test = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), test_idz)))
# in_shp2 = list(Z_train.shape[1:])
# print(Z_train.shape, in_shp2)

#AP数据提取，作为人工特征
filename = './IQ_and_AP_net/A_P_data.pickle'
with open(filename, 'rb') as file:
    M = cPickle.load(file)

#   定义数据集大小
n_examples_2 = M.shape[0]
#   取一半的数据集作为训练集
n_train_2 = n_examples_2 * 0.5
#   选择训练和测试索引
train_idx_2 = np.random.choice(range(0,n_examples_2), size=int(n_train_2), replace=False)
test_idx_2 = list(set(range(0,n_examples_2))-set(train_idx_2))

M_train = M[train_idx_2]
M_test = M[test_idx_2]
#   从训练数据中提取应该输入的形状并打印出来
in_shp_2 = list(M_train.shape[1:])
print(M_train.shape, in_shp_2)



# Build VT-CNN2 Neural Net model using Keras primitives --
#  - Reshape [N,2,128] to [N,2,128,1] on input
#  - Pass through 2 2DConv/ReLu layers
#  - Pass through 2 Dense layers (ReLu and Softmax)
#  - Perform categorical cross entropy optimization
#   构建网络模型，由两个卷积层，两个全连接层，组成，并引入丢弃层防止过拟合
dr = 0.5 # dropout rate(%)丢弃率
input_1 = tf.keras.layers.Input(shape=in_shp_2)
a = tf.keras.layers.Reshape(in_shp_2+[1])(input_1)
a = tf.keras.layers.ZeroPadding2D((0,2), data_format="channels_last")(a)
a = tf.keras.layers.Convolution2D(256, (1, 3),padding='valid', activation="relu", name="conv1", kernel_initializer='glorot_uniform', data_format="channels_last")(a)
a = tf.keras.layers.MaxPooling2D(pool_size=(1, 2), strides=None, padding='valid', data_format=None)(a)
a = tf.keras.layers.ZeroPadding2D((0, 2), data_format="channels_last")(a)
a = tf.keras.layers.Convolution2D(80, (2, 3), padding='valid', activation="relu", name="conv2", kernel_initializer='glorot_uniform', data_format="channels_last")(a)
# a = tf.keras.layers.MaxPooling2D(pool_size=(1, 2), strides=None, padding='valid', data_format=None)(a)
# a = tf.keras.layers.ZeroPadding2D((0, 2), data_format="channels_last")(a)
# a = tf.keras.layers.Convolution2D(80, (1, 3), padding='valid', activation="relu", name="conv3", kernel_initializer='glorot_uniform', data_format="channels_last")(a)
a = tf.keras.layers.MaxPooling2D(pool_size=(1, 6), strides=None, padding='valid', data_format=None)(a)
output_1 = tf.keras.layers.Flatten()(a)
first = tf.keras.Model(input_1, output_1, name="first")
first.summary()

#第二层
input_2 = tf.keras.layers.Input(shape=in_shp)
b = tf.keras.layers.Reshape(in_shp+[1])(input_2)
b = tf.keras.layers.ZeroPadding2D((0, 2), data_format="channels_last")(b)
b = tf.keras.layers.Convolution2D(256, (1, 3),padding='valid', activation="relu", name="conv4", kernel_initializer='glorot_uniform', data_format="channels_last")(b)
b = tf.keras.layers.MaxPooling2D(pool_size=(1, 2), strides=None, padding='valid', data_format=None)(b)
b = tf.keras.layers.ZeroPadding2D((0, 2), data_format="channels_last")(b)
b = tf.keras.layers.Convolution2D(80, (2, 3), padding='valid', activation="relu", name="conv5", kernel_initializer='glorot_uniform', data_format="channels_last")(b)
b = tf.keras.layers.MaxPooling2D(pool_size=(1, 2), strides=None, padding='valid', data_format=None)(b)
b = tf.keras.layers.ZeroPadding2D((0, 2), data_format="channels_last")(b)
# b = tf.keras.layers.Convolution2D(80, (1, 3), padding='valid', activation="relu", name="conv6", kernel_initializer='glorot_uniform', data_format="channels_last")(b)
# b = tf.keras.layers.MaxPooling2D(pool_size=(1, 2), strides=None, padding='valid', data_format=None)(b)
# b = tf.keras.layers.Reshape((17,80),input_shape=(1,17,80))(b)
b = tf.keras.layers.LSTM(100,activation='relu', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', return_sequences=True)(b)
b = tf.keras.layers.LSTM(50, activation='relu', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', return_sequences=True)(b)
output_2 = tf.keras.layers.Flatten()(b)
second = tf.keras.Model(input_2, output_2, name="second")
second.summary()

#两个网络相连接
concate = tf.keras.layers.Concatenate()([output_1, output_2])
print(concate.shape)
concate = tf.keras.layers.Dense(256, activation='relu', kernel_initializer='he_normal', name="dense")(concate)
concate = tf.keras.layers.Dropout(dr)(concate)
print(concate.shape)
concate = tf.keras.layers.Dense(len(classes), kernel_initializer='he_normal', name='dense2')(concate)
concate = tf.keras.layers.Activation('softmax')(concate)
print(concate.shape)
output = tf.keras.layers.Reshape([len(classes)])(concate)
final = tf.keras.Model(inputs=[input_1, input_2], outputs=output)
final.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
final.summary()
# 设置一些参数
nb_epoch = 100  # number of epochs to train on要训练的历元数
batch_size = 512  # training batch size训练批量大小

# perform training ...
#   - call the main training loop in keras for our network+dataset调用keras中主训练循环
# filepath1 = 'convmodrecnets1_CNN2_0.5.wts.h5'
# filepath2 = 'convmodrecnets2_CNN2_0.5.wts.h5'
filepath3 = './IQ_and_AP_net/IQ_and_AP_net.wts.h5'
# Set up some params
history3 = final.fit([M_train, X_train],                         #训练数据
                    Y_train,                         #训练数据对应的标签
                    batch_size=batch_size,           #训练批量大小，每次训练时使用的样本数
                    epochs=nb_epoch,                 #训练轮数，表示模型需要训练的次数
                    #show_accuracy=False,            #表示不显示训练过程中的准确率
                    verbose=2,                       #表示在训练过程中显示详细信息
                    validation_data=([M_test, X_test],Y_test), #使用测试数据X_test和标签Y_text进行模型的验证
                    callbacks = [                    #回调函数，用于在每个训练轮数结束时保存模型的权重
                        tf.keras.callbacks.ModelCheckpoint(filepath3, monitor='val_loss', verbose=0, save_best_only=True, mode='auto'),#filepath是保存模型权重的路径
                        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')#回调函数，用于在验证集上损失函数不再下降时停止训练，5次迭代都没有改进时停止训练
                    ])
print ("训练已经完成，最佳权重模型已保存到根目录！正在加载最佳权重模型....")
final.load_weights(filepath3)
print ("最佳模型加载成功！")
score = final.evaluate([M_test, X_test], Y_test, verbose=0, batch_size=batch_size)
print("Loss: ", score[0])
print("Accuracy: ", score[1])
# with open('history.json', 'w') as f:
#     json.dump(history.history, f)
#     json.dump(history.epoch, f)
#
#   画出损失曲线
plt.figure()
plt.title('Training performance')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.plot(history3.epoch, history3.history['loss'], label='train loss+error')
plt.plot(history3.epoch, history3.history['val_loss'], label='val_error')
plt.grid(True)
plt.legend()
plt.savefig("./IQ_and_AP_net/loss", format='png', dpi=300)


#   定义混淆矩阵，输入为矩阵数据，标题，配色和标签，实际调用时只需要提供混淆矩阵数据和标签列表即可。
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, labels=[]):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.grid(True)

#   批处理对测试集进行预测
test_Y_hat = final.predict([M_test,X_test], batch_size=batch_size)
conf = np.zeros([len(classes),len(classes)])
confnorm = np.zeros([len(classes),len(classes)])
#   遍历所有测试样本
for i in range(0,X_test.shape[0]):
    #   找到第i个测试样本的标签，因为是二进制编码格式。所以找到的是1的索引
    j = list(Y_test[i,:]).index(1)
    #   找到预测结果中概率最大的元素索引
    k = int(np.argmax(test_Y_hat[i,:]))
    #   对应混淆矩阵上的点加1
    conf[j,k] = conf[j,k] + 1
#   遍历所有类别
for i in range(0,len(classes)):
    #   对原始混淆矩阵进行归一化
    confnorm[i,:] = conf[i,:] / np.sum(conf[i,:])
plot_confusion_matrix(confnorm, labels=classes)
plt.savefig("./IQ_and_AP_net/confu_matrix_total",format='png', dpi=300)
acc = {}
#   取出测试集的所有信噪比列表，根据测试集索引test_idx来取
test_SNRs = list(map(lambda x: lbl[x][1], test_idx))

#   把指定信噪比下的测试数据提取出来
for snr in snrs:
    #   作判断是不是指定信噪比
    snr_bool = np.array(test_SNRs) == snr
    #   找到匹配的信噪比索引
    snr_idx = np.where(snr_bool)
    #   取出该信噪比下的数据
    test_X_i = X_test[snr_idx]
    test_M_i = M_test[snr_idx]
    #   取出标签
    test_Y_i = Y_test[snr_idx]
    print(len(snr_idx[0]))
    # estimate classes对该信噪比的测试集数据进行预测
    test_Y_i_hat = final.predict([test_M_i,test_X_i], batch_size=batch_size)
    #   初始化混淆矩阵
    conf1 = np.zeros([len(classes), len(classes)])
    confnorm1 = np.zeros([len(classes), len(classes)])
    #   遍历测试样本，构建原始混淆矩阵
    for i in range(0,test_X_i.shape[0]):
        j = list(test_Y_i[i,:]).index(1)
        k = int(np.argmax(test_Y_i_hat[i,:]))
        conf1[j,k] = conf1[j,k] + 1
    #   归一化混淆矩阵
    for i in range(0,len(classes)):
        confnorm1[i,:] = conf1[i,:] / np.sum(conf1[i,:])
    plt.figure()
    plot_confusion_matrix(confnorm1, labels=classes, title="ConvNet Confusion Matrix (SNR=%d)"%(snr))
    plt.savefig("./IQ_and_AP_net/comf_Matrix_for_snr=" + str(snr), format='png', dpi=300)  # 设置 dpi 参数以调整保存的图像质量
    #   拿到原始混淆矩阵对角线的元素并求和
    cor = np.sum(np.diag(conf1))
    #   求出除了对角线元素外的所有元素的和
    ncor = np.sum(conf1) - cor
    #   总体准确率为预测对的数量比上总数
    print("Overall Accuracy: ", cor / (cor+ncor))
    acc[snr] = 1.0*cor/(cor+ncor)
# Save results to a pickle file for plotting later
print(acc)
fd = open('./IQ_and_AP_net/acc_trend.dat', 'wb')
cPickle.dump( ("CNN2", 0.5, acc) , fd )
 # Plot accuracy curve
plt.figure()
plt.plot(snrs, list(map(lambda x: acc[x], snrs)))
plt.xlabel("Signal to Noise Ratio")
plt.ylabel("Classification Accuracy")
plt.title("acc_trend for double_net")
plt.grid(True)
plt.savefig("./IQ_and_AP_net/acc_trend", format='png', dpi=300)  # 设置 dpi 参数以调整保存的图像质量