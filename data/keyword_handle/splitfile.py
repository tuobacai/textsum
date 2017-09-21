# coding:utf-8
import codecs
def split():
    file='/Users/yingjie10/infoextract/seq2seq/data/eventfile.txt'
    trains='/Users/yingjie10/infoextract/seq2seq/data/keyword/train_content.txt'
    trainss = '/Users/yingjie10/infoextract/seq2seq/data/keyword/train_title.txt'
    devs = '/Users/yingjie10/infoextract/seq2seq/data/keyword/dev_content.txt'
    devss = '/Users/yingjie10/infoextract/seq2seq/data/keyword/dev_title.txt'
    tests = '/Users/yingjie10/infoextract/seq2seq/data/keyword/test_content.txt'
    testss = '/Users/yingjie10/infoextract/seq2seq/data/keyword/test_title.txt'

    tfile=open(trains,'w')
    tsfile = open(trainss, 'w')

    dfile = open(devs, 'w')
    dsfile = open(devss, 'w')

    sfile = open(tests, 'w')
    ssfile = open(testss, 'w')

    f=open(file,'r')
    cnt=0
    for line in f:
        lines=line.strip().split(',')
        content=lines[0]
        title=lines[1]
        if cnt<400:
            tfile.write(content+'\n')
            tsfile.write(title+'\n')
        if cnt>=400 and cnt <600:
            tfile.close()
            tsfile.close()
            dfile.write(content + '\n')
            dsfile.write(title + '\n')
        if cnt>=600:
            dfile.close()
            dsfile.close()
            sfile.write(content + '\n')
            ssfile.write(title + '\n')
        cnt+=1
    sfile.close()
    ssfile.close()
'''
eventfile.txt: wordcnt,243242;len(words),31738

'''
def statistic():
    file = '/Users/yingjie10/textsum/data/keyword_handle/eventfile.txt'
    trains = '/Users/yingjie10/infoextract/seq2seq/data/keyword/train_content.txt'
    trainss = '/Users/yingjie10/infoextract/seq2seq/data/keyword/train_title.txt'
    devs = '/Users/yingjie10/infoextract/seq2seq/data/keyword/dev_content.txt'
    devss = '/Users/yingjie10/infoextract/seq2seq/data/keyword/dev_title.txt'
    tests = '/Users/yingjie10/infoextract/seq2seq/data/keyword/test_content.txt'
    testss = '/Users/yingjie10/infoextract/seq2seq/data/keyword/test_title.txt'

    # tfile = open(trains, 'r')
    # tsfile = open(trainss, 'r')
    #
    # dfile = open(devs, 'r')
    # dsfile = open(devss, 'r')
    #
    # sfile = open(tests, 'r')
    # ssfile = open(testss, 'r')

    ''' 统计大文件单词指标'''
    f = codecs.open(file, 'r',encoding='utf-8')
    words=dict()
    cnt=0
    wordcnt=0
    for line in f:
        cnt=cnt+1
        lines = line.strip().split(',')
        content = lines[0]
        ws=content.split(' ')
        for w in ws:
            if cnt==1:
                print 'w:',w
            wordcnt+=1
            if w not in words:
                words[w]=1
    f.close()
    print len(words),wordcnt


statistic()