from database import dataInit
from train import setOfWord2Vec
from LinearModel import my_linear
import numpy as np
import torch
import lib
import csv

def test(p0Vec,p1Vec,p1_class,vocab_list):
    test_text_list, test_label_list, test_vocab_list = dataInit(Train=False, Output=False)
    sumNumText = len(test_label_list)  # test集总句子数
    trainSet_test = list(range(sumNumText))
    data_test = []
    for idx in trainSet_test:
        wordVec = setOfWord2Vec(vocab_list, test_text_list[idx])
        p0_n = [np.log(1-p1_class)/100]*lib.max_word_len
        p0 = np.array(wordVec) * np.array(p0Vec) + p0_n
        p1_n =  [np.log(p1_class)/100]*lib.max_word_len
        p1 = np.array(wordVec) * np.array(p1Vec) + p1_n
        wordVec = p0 * p1_class + p1 * (1 - p1_class)
        data_test += [[wordVec,test_label_list[idx]]]  # 预测结果
    with torch.no_grad():
        errorCount_zero = 0
        zero = 0
        errorCount_one = 0
        one = 0
        for idx in trainSet_test:
            input = torch.Tensor(data_test[idx][0])
            output = my_linear(input)
            target = data_test[idx][1]
            predict = 1 if output.item() >= lib.threshold else 0
            print(idx, output.item(),predict,target)
            if target == 0:
                zero += 1
                if predict != target:
                    errorCount_zero += 1
            else:
                one += 1
                if predict != target:
                    errorCount_one += 1
        print("successRate:", 1.0 - errorCount_zero / zero, 1.0 - errorCount_one / one,
              (2.0 - errorCount_zero / zero - errorCount_one / one) * 0.5)

def test_output(p0Vec,p1Vec,p1_class,vocab_list):
    test_text_list, test_label_list, test_vocab_list = dataInit(Train=False, Output=True)
    sumNumText = len(test_label_list)  # test集总句子数
    trainSet_test = list(range(sumNumText))
    data_test = []
    for idx in trainSet_test:
        wordVec = setOfWord2Vec(vocab_list, test_text_list[idx])
        p0_n = [np.log(1 - p1_class) / 100] * lib.max_word_len
        p0 = np.array(wordVec) * np.array(p0Vec) + p0_n
        p1_n = [np.log(p1_class) / 100] * lib.max_word_len
        p1 = np.array(wordVec) * np.array(p1Vec) + p1_n
        wordVec = p0 * p1_class + p1 * (1 - p1_class)
        data_test += [[wordVec]]  # 预测结果

    with torch.no_grad():
        ans_list = []
        for idx in trainSet_test:
            input = torch.Tensor(data_test[idx][0])
            output = my_linear(input)
            predict = 1 if output.item() >= lib.threshold else 0
            print(idx + 1,output.item(),predict)
            ans_list += [[idx + 1, predict]]
        print(ans_list)
        with open(lib.output_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'label'])
            writer.writerows(ans_list)
        with open(lib.output_path, 'rt') as fin:  # 读有空行的csv文件，舍弃空行
            lines = ''
            for line in fin:
                if line != '\n':
                    lines += line
        with open(lib.output_path, 'wt') as fout:  # 再次文本方式写入，不含空行
            fout.write(lines)
