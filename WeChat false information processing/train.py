import numpy as np
import lib

def setOfWord2Vec(vocab_list,content):
    returnVec = [0] * lib.max_word_len
    for word in content:
        if word in vocab_list:
            returnVec[vocab_list.index(word)] = 1 #将出现过词的位置置为1
    return returnVec

def trainNB(trainMat,trainLabel,epoch): # 1表示假，0表示真
    numTrain = len(trainMat)
    numWords = len(trainMat[0])
    p1 = sum(trainLabel)/float(numTrain) #假新闻概率 = 假新闻总数 / 总新闻个数
    p0Num = np.ones(numWords)  # 平滑处理，防止概率值为0
    p1Num = np.ones(numWords)
    p0Denom = 2  # 分母初始化成类别数，分母为词的总数
    p1Denom = 2
    for idx in range(numTrain):
        if trainLabel[idx] == 1:
            p1Num +=  lib.fake_weight * trainMat[idx] * epoch #乘上权重,统计词频
            p1Denom += sum(trainMat[idx]) #加上词的总数
        else:
            p0Num += trainMat[idx] * epoch
            p0Denom += sum(trainMat[idx])  # 加上词的总数
    p1Vec = np.log(p1Num / p1Denom / epoch)
    p0Vec = np.log(p0Num / p0Denom / epoch)
    return p0Vec,p1Vec,p1

def BayesDataProcess(train_text_list, train_label_list, train_vocab_list):
    trainSet = list(range(len(train_label_list)))
    np.random.shuffle(trainSet)
    trainMat = []
    trainLabel = []
    tmp = 0
    for index in trainSet:
        tmp += 1
        print(tmp)
        trainMat.append(setOfWord2Vec(train_vocab_list, train_text_list[index]))  # 把文字向量化
        trainLabel.append(train_label_list[index])
    p0Vec, p1Vec, p1_class = trainNB(np.array(trainMat), np.array(trainLabel), lib.epoch)
    data = []
    for idx in range(len(trainLabel)):
        p0_n = [np.log(p1_class) / 100] * lib.max_word_len
        p0 = np.array(trainMat[idx]) * np.array(p0Vec) + p0_n
        p1_n = [np.log(1 - p1_class) / 100] * lib.max_word_len
        p1 = np.array(trainMat[idx]) * np.array(p1Vec) + p1_n
        wordVec = p0 * p1_class + p1 * (1 - p1_class)
        data += [[wordVec, trainLabel[idx]]]  # 预测结果
    return p0Vec, p1Vec, p1_class,data