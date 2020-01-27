from utils import load,get_batch,predict_from_nothing,predict_from_head
from keras.models import Model
from keras.layers import CuDNNLSTM,Dense,Input,Softmax,Convolution1D,Embedding,Dropout
UNITS = 256
poetry_file = 'poetry.txt'

# 载入数据
x_data,char2id_dict,id2char_dict = load(poetry_file)
max_length = max([len(txt) for txt in x_data])
words_size = len(char2id_dict)

#-------------------------------#
#   建立神经网络
#-------------------------------#
inputs = Input(shape=(None,words_size))
x = CuDNNLSTM(UNITS,return_sequences=True)(inputs)
x = Dropout(0.6)(x)
x = CuDNNLSTM(UNITS)(x)                  
x = Dropout(0.6)(x)
x = Dense(words_size, activation='softmax')(x)
model = Model(inputs,x)

model.load_weights("logs/loss4.419-val_loss4.009.h5")
predict_from_nothing(0,x_data,char2id_dict,id2char_dict,model)
predict_from_head("快乐",x_data,char2id_dict,id2char_dict,model)