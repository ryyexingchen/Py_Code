import re
import jieba
from jieba import analyse
import pickle
import lib
import os
import csv
def tokenlize(content,Train=True,Output=False): #分词
    content_text = re.sub(r"\s+",' ',content[0],flags=re.S)
    # content = re.split(',',content)
    # if Train:
    #     label = int(content[-1])
    #     content_text = content[1]
    # else:
    #     if Output:
    #         label = -1
    #         content_text = content[2]
    #     else:
    #         label = int(content[-1])
    #         content_text = content[1]

    filters = ['【', '】', '“', '”', '‘', '’', '~', '\\t', '\\n', '\\x97', '\\x96', '：', '；', '!', '"', '#', '\\s', '$','%', '&', '\\(', '\\)', '\\*', '\\+', ',', '-', '\\.', '/', ':', ';', '<', '=', '>','\\?', '@', '\\[', '\\\\', '\\]', '##', '^', '_', '、', '▎', '`', '\\{', '\\|', '\\}', '？', '！', '——','，', '。', '（', '）', '…', '《', '》','●', '㊙', '▶', '�', '·', '°', '℃','✅']  # 需要删除的无用字符
    content_text = re.sub('|'.join(filters),'',content_text,flags=re.S)
    if Train:
        tfidf = analyse.extract_tags
        tokens = tfidf(content_text)
        # tokens = jieba.lcut(content_text,cut_all=True)
        print(tokens)
    else:
        tfidf = analyse.extract_tags
        tokens = tfidf(content_text)
        print(tokens)
    # if Train and Output:
    #     tokens += content[1] # 加入新闻媒体名，不分词
    # else:
    #     tokens += content[0]
    if Train:
        label = int(content[-1])
    else:
        label = -1
    return tokens,label

def createVocablist(doc_list): #创建语料表
    vocabSet = set([])
    for word in doc_list:
        vocabSet = vocabSet|set(word)
    return list(vocabSet)


def dataInit(Train=True,Output=False): #初始化数据（词典）
    text_list = [] #存放所有文本
    label_list = [] #存放所有标签
    if Train and os.path.exists(lib.train_text_path): # 三个文件是一起保存的,如果之前保存过就不需要再保存了
        with open(lib.train_text_path, 'rb') as f:
            text_list = pickle.load(f)
        with open(lib.train_label_path, 'rb') as f:
            label_list = pickle.load(f)
        with open(lib.train_vocab_path, 'rb') as f:
            vocab_list = pickle.load(f)
        return text_list,label_list,vocab_list
    else:
        if Train: # 读取路径
            data_path = lib.train_path
        else:
            if Output:
                data_path = lib.test_path_output
            else:
                data_path = lib.test_path

        # content = open(data_path,encoding="utf-8").read().replace(' ','')
        # content_list = list(re.split('\n',content))
        # del content_list[0]
        # del content_list[-1] # 将每一句话拆分成一个列表
        reading_way = "utf-8"
        # if not Train:
        #     reading_way = "ANSI"
        reader_list = list(csv.reader(open(data_path,encoding=reading_way)))
        del reader_list[0]
        content_list = []
        for row in reader_list:
            if Train:
                content_list += [[row[1],int(row[-1])]]
            else:
                content_list += [[row[1]]]
        print(content_list)
        for content in content_list:
            word_list,label = tokenlize(content,Train,Output) # 将每一句话进行分词
            text_list.append(word_list)
            label_list.append(label)

        vocab_list = createVocablist(text_list) #语料表

        if Train:
            with open(lib.train_text_path, 'wb') as f:
                pickle.dump(text_list, f)
            with open(lib.train_label_path, 'wb') as f:
                pickle.dump(label_list, f)
            with open(lib.train_vocab_path, 'wb') as f:
                pickle.dump(vocab_list, f)
        return text_list,label_list,vocab_list

if __name__ == '__main__':
    # dataInit(False)
    pass