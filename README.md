# 2D-Anime-Image-Detecting-with-Machine-Learning
System: Windows 10   
Develope Tool: Anaconda, Visual Studio Code  
Python Version: 3.6  
Packages: tensorflow 1.12.0, tensorflow-base 1.12.0, tensorboard 1.12.2  

How to Execute:  
Before executing any python file, please downloard the raw data "all_data.csv"  
https://drive.google.com/file/d/1_NI03b-k1tBfXllhDfpH6PAlPKyl9GXx/view?usp=sharing  
  
If you want to do a fast execute without downloading data using python programs. Please download the rest of data from google drive links:  
download_images.zip (whole image dataset for training):  
https://drive.google.com/file/d/1mh2GsqmpbK9OYjjspCi7oYoUa6c79utK/view?usp=sharing  
anime_images.zip (actually used in training):  
https://drive.google.com/file/d/1JJLjCw-P5fE6YSKix7H7Hkd1_wQo1W3k/view?usp=sharing  
test_images.zip (whole image dataset for testing):  
https://drive.google.com/file/d/1uxMRIeHG-C4kk81jGfix4qMqDxV0KY7C/view?usp=sharing  
(Links of all those datas are also stored in "dataLink.txt")  
  
If you want to execute without the database I have already generated, please follow next few steps:  
 1. Create a folder named "anime_images" and "images" under project folder.  
 2. Run "download.py".  
 3. Run "deal.py".  
 4. Run "check.py".  
 5. Run "download_test.py".  
  
 For training data, run "train.py". The training result will be saved in "train_saver" folder and the training record will be stored in "training_output.txt".  
   
 For testing data, run "test.py". The testing result will be printed on the terminal.  
 Currently the code for advanced testing (testing images of out database and adding tags on image) have been commented. If you want to try the advanced testing, please follow next few steps:  
  1. Create a new folder called "test_image_resize", put the image files you want to test into this folder.  
  2. In "test.py", uncomment from line 70 to 78 and line 144 to 157.  
  3. In "test.py", comment from line 46 to 69 and line 140 to 143.  
  4. In "test.py", change the range of i in line 72 with the number of your test images.  
After doing this modification, you can run "test.py" and the new image with tag will be generated under the folder "test_image_resize".  
