import urllib
import urllib.request
import csv
import os
import fileinput
import PIL
from PIL import Image

dic = []
for line in fileinput.input('dic.txt'):
    os.mkdir('anime_images/'+ str(line).strip('\n') + '/')
    dic.append(str(line).strip('\n'))

csv_file=open('all_data.csv',"r",encoding='utf-8')
csv_reader_lines = csv.reader(csv_file)
num = 0


for one_line in csv_reader_lines:
    if num % 1000 == 0:
        print(num)
    if num < 55000:
        try:
            img = Image.open('images/'+one_line[0]+'.JPEG')
            elementList =  one_line[8].split()
            for i in elementList:
                if i in dic:
                    img.save('anime_images/' + i + '/' +one_line[0]+'.jpg')
        except:
            print("download error with ", one_line[0]+'.JPEG')
        num += 1
    else :
        break