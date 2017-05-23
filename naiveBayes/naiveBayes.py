import csv
import math

def getData(csv_file):
    csvfile = file(csv_file,'rb')
    reader = csv.DictReader(csvfile)
    data_list = [row for row in reader]
    csvfile.close()
    return data_list

def getProbability(train_data):
    condi_prob = {}
    N = 2  #class number: spam or non_spam
    p = 4; #divide into p parts
    is_spam = [int(i['is_spam']) for i in train_data]
    total_num, spam_num = float(len(is_spam)), float(sum(is_spam))
    good_num = total_num - spam_num
    condi_prob['prior'] = [(good_num + 1)/(total_num + N), (spam_num + 1)/(total_num + N)]

    attrs = [key for key in train_data[0]]  # attributes in data
    for attr in attrs:
        if attr=='is_spam':
            continue
        if attr=='capital_run_length_average' or attr=='capital_run_length_longest' or attr=='capital_run_length_total':
            record = [float(row[attr]) for row in train_data]
            maxR, minR = max(record), min(record)
            width = (maxR + 0.1 - minR) / p
            # divide the data into p interval
            partition = [minR + i * width for i in range(1,p+1)]
            count = [[0 for i in range(p)] for j in range(2)]
            cnt = 0
            for r in record:
                index = int(math.floor((r - minR) / width))
                if (index >= p):
                    index = p - 1
                type_inx = 0
                if (is_spam[cnt] > 0):
                    type_inx = 1
                count[type_inx][index] += 1
                cnt += 1

            for i in range(p):
                count[0][i] = float(count[0][i] + 1) / (good_num + p)
                count[1][i] = float(count[1][i] + 1) / (spam_num + p)

            condi_prob[attr] = [count[0], count[1],partition]

        else:
            record = [float(row[attr]) for row in train_data]
            count = [[0 for i in range(2)] for j in range(2)]
            cnt = 0
            for r in record:
                type_inx = 0
                if (is_spam[cnt] > 0):
                    type_inx = 1
                if_appear = 0;
                if (r>0):
                    if_appear = 1;
                count[type_inx][if_appear] += 1
                cnt += 1

            for i in range(2):
                count[0][i] = float(count[0][i] + 1) / (good_num + 2)
                count[1][i] = float(count[1][i] + 1) / (spam_num + 2)

            condi_prob[attr] = [count[0], count[1]]

    return condi_prob



def classify(test_file,condi_prob):
    test_data = getData(test_file)
    #test_data=test_data[0:10]
    cnt=0

    total=len(test_data)
    wrong = 0
    equal = 0
    for row in test_data:
        prob = [0,0]
        prob[0]=condi_prob['prior'][0]
        prob[1]=condi_prob['prior'][1]
        res = int(row['is_spam'])
        for key in condi_prob:
            if key=='prior':
                continue
            if key == 'capital_run_length_average' or key == 'capital_run_length_longest' or key == 'capital_run_length_total':
                test_inx = find_inx(float(row[key]), condi_prob[key][2])
                #print test_inx
                prob[0]*=condi_prob[key][0][test_inx]
                prob[1]*=condi_prob[key][1][test_inx]
                continue
            else:
                if_exist = 0
                if (float(row[key])>0):
                    if_exist = 1

                prob[0] *= condi_prob[key][0][if_exist]
                prob[1] *= condi_prob[key][1][if_exist]

        predict = 0
        if prob[1]>prob[0]:
            cnt+=1
            predict=1
        if predict!=res:
            wrong+=1
        if prob[0]==prob[1]:
            equal+=1

        accuracy = 100-float(wrong)/total *100
    print accuracy
    #print cnt,wrong,total,equal


def find_inx(value,partition):
    l = len(partition)
    for i in range(l):
        if(value<=partition[i]):
            return i
    return l-1


def main():
    train_data = getData('train.csv')
    condi_prob = getProbability(train_data)
    classify('test.csv',condi_prob)

if __name__ =="__main__":
    main()
