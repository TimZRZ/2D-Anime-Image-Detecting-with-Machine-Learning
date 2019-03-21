import urllib
import urllib.request
import csv
import os
import fileinput

csv_file=open('all_data.csv',"r",encoding='utf-8')
csv_reader_lines = csv.reader(csv_file)
num = 0

for one_line in csv_reader_lines:
    if num % 1000 == 0:
        print(num)
    if num < 100000:
        try:
            img_url = str(one_line[7])
            urllib.request.urlretrieve(img_url,'images/'+one_line[0]+'.JPEG')
        except:
            print("download error with ", one_line[0]+'.JPEG')
    num += 1
    
