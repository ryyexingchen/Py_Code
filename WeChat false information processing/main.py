import lib
from database import dataInit
from train import BayesDataProcess
from LinearModel import linear_train
from test import test_output,test

def find_in_train(train_text_list,train_label_list,output_path,Output=False):
    test_text_list,test_label_list,text_vocal_list = dataInit(Train=False,Output=Output)
    sumNumText = len(test_label_list)  # test集总句子数
    trainSet_test = list(range(sumNumText))

def trainProcess(epoch):
    train_text_list, train_label_list, train_vocab_list = dataInit()
    p0Vec, p1Vec, p1,data = BayesDataProcess(train_text_list, train_label_list, train_vocab_list)
    # linear_train(data,epoch)
    testProcess(p0Vec, p1Vec, p1,train_vocab_list)

def testProcess(p0Vec, p1Vec, p1_class,train_vocab_list):
    test_output(p0Vec, p1Vec, p1_class, train_vocab_list)
    # test(p0Vec, p1Vec, p1_class, train_vocab_list)

if __name__ == '__main__':
    trainProcess(lib.linear_epoch)
