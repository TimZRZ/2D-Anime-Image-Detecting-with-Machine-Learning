import urllib
import urllib.request
import csv

csv_file=open('all_data.csv',"r",encoding='utf-8')
csv_reader_lines = csv.reader(csv_file)
dic = {}
num = 0
fileObject = open('dic.txt', 'w',encoding='utf-8')

for one_line in csv_reader_lines:
    if num != 0:
        if num % 1000000 == 0:
            print(num)
        try:
            elementList =  one_line[8].split()
            for i in elementList:   
                if i not in dic.keys():
                    dic[i] = 1
                else:
                    dic[i] += 1
        except:
            print("error ", num)
    num += 1
print(len(dic.keys()))
for i in dic.keys():
    #if dic[i] > 50000:
    if dic[i] > 120000:
        fileObject.write(str(i))
        fileObject.write('\n')

fileObject.close()