import numpy as np
import collections

END_CHAR = '\n'
UNKNOWN_CHAR = ' '
unit_sentence = 6
max_words = 3000
MIN_LENGTH = 10

def load(poetry_file):
    def handle(line):
        return line + END_CHAR

    poetrys = [line.strip().replace(' ', '').split(':')[1] for line in
                    open(poetry_file, encoding='utf-8')]
    collect = []
    for poetry in poetrys:
        if len(poetry) <= 5 :
            continue
        if poetry[5]=="，":
            collect.append(handle(poetry))
    print(len(collect))
    poetrys = collect
    # 所有字
    words = []
    for poetry in poetrys:
        words += [word for word in poetry]
    counter = collections.Counter(words)
    
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])
    # 获得所有字，出现次数从大到小排列

    words, _ = zip(*count_pairs)
    # 取出现频率最高的词的数量组成字典，不在字典中的字用'*'代替
    words_size = min(max_words, len(words))
    words = words[:words_size] + (UNKNOWN_CHAR,)
    # 计算总长度
    words_size = len(words)

    # 字映射成id，采用ont-hot的形式
    char2id_dict = {w: i for i, w in enumerate(words)}
    id2char_dict = {i: w for i, w in enumerate(words)}
    
    unknow_char = char2id_dict[UNKNOWN_CHAR]
    char2id = lambda char: char2id_dict.get(char, unknow_char)
    poetrys = sorted(poetrys, key=lambda line: len(line))
    # 训练集中每一首诗都找到了每个字对应的id
    poetrys_vector = [list(map(char2id, poetry)) for poetry in poetrys]
    return np.array(poetrys_vector),char2id_dict,id2char_dict

def get_6to1(x_data,char2id_dict):
    inputs = []
    targets = []
    for index in range(len(x_data)):
        x = x_data[index:(index+unit_sentence)]
        y = x_data[index+unit_sentence]
        if (END_CHAR in x) or y == char2id_dict[END_CHAR]:
            return np.array(inputs),np.array(targets)
        else:
            inputs.append(x)
            targets.append(y)
    return np.array(inputs),np.array(targets)

def get_batch(batch_size,x_data,char2id_dict,id2char_dict):
    
    n = len(x_data)

    batch_i = 0

    words_size = len(char2id_dict)
    while(True):
        one_hot_x_data = []
        one_hot_y_data = []
        for i in range(batch_size):
            batch_i = (batch_i+1)%n
            inputs,targets = get_6to1(x_data[batch_i],char2id_dict)
            for j in range(len(inputs)):
                one_hot_x_data.append(inputs[j])
                one_hot_y_data.append(targets[j])
            
        batch_size_after = len(one_hot_x_data)
        input_data = np.zeros(
            (batch_size_after, unit_sentence, words_size))
        target_data = np.zeros(
            (batch_size_after, words_size))
        for i, (input_text, target_text) in enumerate(zip(one_hot_x_data, one_hot_y_data)):
            # 为末尾加上" "空格
            for t, index in enumerate(input_text):
                input_data[i, t, index] = 1
            
            # 相当于前一个内容的识别结果，作为输入，传入到解码网络中
            target_data[i, target_text] = 1.
        yield input_data,target_data

def predict_from_nothing(epoch,x_data,char2id_dict,id2char_dict,model):
    # 训练过程中，每1个epoch打印出当前的学习情况
    print("\n#-----------------------Epoch {}-----------------------#".format(epoch))
    words_size = len(id2char_dict)
    
    index = np.random.randint(0, len(x_data))
    sentence = x_data[index][:unit_sentence]
    def _pred(text):
        temp = text[-unit_sentence:]
        x_pred = np.zeros((1, unit_sentence, words_size))
        for t, index in enumerate(temp):
            x_pred[0, t, index] = 1.
        preds = model.predict(x_pred)[0]
        choice_id = np.random.choice(range(len(preds)),1,p=preds)
        if id2char_dict[choice_id[0]] == ' ':
            while id2char_dict[choice_id[0]] in ['，','。',' ']:
                choice_id = np.random.randint(0,len(char2id_dict),1)
        return choice_id

    for i in range(24-unit_sentence):
        pred = _pred(sentence)
        sentence = np.append(sentence,pred)
    output = ""
    for i in range(len(sentence)):
        output = output + id2char_dict[sentence[i]]
    print(output)

def predict_from_head(epoch,name,x_data,char2id_dict,id2char_dict,model):
    # 根据给定的字，生成藏头诗
    if len(name) < 4:
        for i in range(4-len(name)):
            index = np.random.randint(0,len(char2id_dict))
            while id2char_dict[index] in ['，','。',' ']:
                index = np.random.randint(0,len(char2id_dict))
            name += id2char_dict[index]

    origin_name = name
    name = list(name)

    for i in range(len(name)):
        if name[i] not in char2id_dict:
            index = np.random.randint(0,len(char2id_dict))
            while id2char_dict[index] in ['，','。',' ']:
                index = np.random.randint(0,len(char2id_dict))
            name[i] = id2char_dict[index]

    name = ''.join(name)
    words_size = len(char2id_dict)
    index = np.random.randint(0, len(x_data))

    #选取随机一首诗的最后max_len字符+给出的首个文字作为初始输入
    sentence = np.append(x_data[index][-unit_sentence:-1],char2id_dict[name[0]])

    def _pred(text):
        temp = text[-unit_sentence:]
        x_pred = np.zeros((1, unit_sentence, words_size))
        for t, index in enumerate(temp):
            x_pred[0, t, index] = 1.
        preds = model.predict(x_pred)[0]
        choice_id = np.random.choice(range(len(preds)),1,p=preds)
        if id2char_dict[choice_id[0]] == ' ':
            while id2char_dict[choice_id[0]] in ['，','。',' ']:
                choice_id = np.random.randint(0,len(char2id_dict),1)
        return choice_id

    for i in range(5):
        pred = _pred(sentence)
        sentence = np.append(sentence,pred)

    sentence = sentence[-unit_sentence:]
    for i in range(3):
        sentence = np.append(sentence,char2id_dict[name[i+1]])
        for i in range(5):
            pred = _pred(sentence)
            sentence = np.append(sentence,pred)

    output = []
    for i in range(len(sentence)):
        output.append(id2char_dict[sentence[i]])
    for i in range(4):
        output[i*6] = origin_name[i]
    output = ''.join(output)

    print(output)