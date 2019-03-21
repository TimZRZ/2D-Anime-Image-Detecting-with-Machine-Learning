from skimage import io,transform
import tensorflow as tf
import numpy as np
import fileinput
import csv
import matplotlib.mlab as mlab  
import matplotlib.pyplot as plt 
from PIL import Image,ImageDraw,ImageFont


#path1 = "test_images_2/2.jpg"


tag_dict = {}
tag_list = []


for line in fileinput.input('dic.txt'):
    tag_list.append(str(line).strip('\n'))
tag_list.sort()

num = 0
for tag in tag_list:
    tag_dict[num] = tag
    num += 1

w=100
h=100
c=3

images_output_list = []

def read_one_image(path):
    img = io.imread(path)
    images_output_list.append(img)
    img = transform.resize(img,(w,h))
    return np.asarray(img)

csv_file=open('all_data.csv',"r",encoding='utf-8')
csv_reader_lines = csv.reader(csv_file)
num = 0

data = []
image_name = []
image_dict = {}

for one_line in csv_reader_lines:
    if num < 1000:
        try:
            path1 = "test_images/" + str(one_line[0]) +".jpg"
            elementList =  one_line[8].split()
            cout = 0
            for k in elementList:
                if k in tag_dict.values():
                    cout += 1
                if cout >= 15:
                    data1 = read_one_image(path1)
                    image_name.append(str(one_line[0]))
                    image_dict[str(one_line[0])] = elementList
                    data.append(data1)
                    num += 1
                    break
            else:
                continue
        except:
            pass
    else :
        break

'''

for i in range(3):
    path1 = "test_image_resize/" + str(i) +".jpg"
    data1 = read_one_image(path1)
    image_name.append(str(i))
    data.append(data1)
    num += 1
'''

print("Get " + str(num) + "data in total")

with tf.Session() as sess:
    saver = tf.train.import_meta_graph('train_saver/model.ckpt.meta')
    saver.restore(sess,tf.train.latest_checkpoint('train_saver/'))

    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0")
    feed_dict = {x:data}

    logits = graph.get_tensor_by_name("logits_eval:0")

    classification_result = sess.run(logits,feed_dict)

    max1 = 0
    max2 = 0
    max3 = 0
    max4 = 0
    max5 = 0
    max6 = 0
    max1_index = 0
    max2_index = 0
    max3_index = 0
    max4_index = 0
    max5_index = 0
    max6_index = 0

    t_acc = 0.0

    for i in range(num) :
        acc = 0.0
        for j in range(38):
            if classification_result[i][j] > max1:
                max1 = classification_result[i][j]
                max1_index = j
                continue
            if (classification_result[i][j]>max2) and (classification_result[i][j]<=max1):
                max2 = classification_result[i][j]
                max2_index = j
                continue
            if (classification_result[i][j]>max3) and (classification_result[i][j]<=max2):
                max3 = classification_result[i][j]
                max3_index = j
                continue
            if (classification_result[i][j]>max4) and (classification_result[i][j]<=max3):
                max4 = classification_result[i][j]
                max4_index = j
                continue
            if (classification_result[i][j]>max5) and (classification_result[i][j]<=max4):
                max5 = classification_result[i][j]
                max5_index = j
                continue
            if classification_result[i][j] > max6 and (classification_result[i][j]<=max5):
                max6 = classification_result[i][j]
                max6_index = j
                continue
        
        for k in [max1_index, max2_index, max3_index, max4_index, max5_index, max6_index]:
            if tag_dict[k] in image_dict[image_name[i]]:
                acc += 0.25
        
        print("ID:" + image_name[i]+" | " + 
            "ACC: " + str(acc)+" | " +"tags: "+tag_dict[max1_index]+', '+tag_dict[max2_index]+ ', ' +tag_dict[max3_index] +', '+tag_dict[max4_index]+ ', ' +tag_dict[max5_index])
        t_acc += acc / num
        '''
        print("ID:" + image_name[i]+" | " +"tags: "+tag_dict[max1_index]+', '+tag_dict[max2_index]+ ', ' +tag_dict[max3_index] +', '+tag_dict[max4_index]+ ', ' +tag_dict[max5_index]+ ', ' +tag_dict[max6_index])
        im1 = Image.open("test_image_resize/" + str(i) +".jpg")
        draw = ImageDraw.Draw(im1)
        font = ImageFont.truetype("C:\Windows\Fonts\Arial.ttf", 28)
        draw.text((10, 10), tag_dict[max1_index], (255, 0, 0), font=font)    
        draw.text((10, 40), tag_dict[max2_index], (255, 0, 0), font=font)    
        draw.text((10, 70), tag_dict[max3_index], (255, 0, 0), font=font)    
        draw.text((10, 100), tag_dict[max4_index], (255, 0, 0), font=font)  
        draw.text((10, 130), tag_dict[max5_index], (255, 0, 0), font=font) 
        draw.text((10, 170), tag_dict[max6_index], (255, 0, 0), font=font)  
        draw = ImageDraw.Draw(im1)
        im1.save("output"+str(i)+".jpg")
        '''
        max1 = 0
        max2 = 0
        max3 = 0
        max4 = 0
        max5 = 0
        max6 = 0
        max1_index = 0
        max2_index = 0
        max3_index = 0
        max4_index = 0
        max5_index = 0
        max6_index = 0
    print("Total acc: " + str(t_acc))

    labels=['Correct','Incorrect']
    X=[t_acc,1-t_acc]  
    
    fig = plt.figure()
    plt.pie(X,labels=labels,autopct='%1.2f%%')
    plt.title("Correction Pie Chart")
    
    plt.show()  
