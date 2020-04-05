#Model Name - CRTXNET
#Model Description - Tracks the ball in each frame of a video

import glob
import xml.etree.ElementTree as ET
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
import itertools
import argparse
from PIL import Image
from keras.models import *
from keras.layers import *
from keras import optimizers
from keras.utils import plot_model
from collections import defaultdict

###Global Declaration###
size = 50
variance = 50
img_w = 3840
img_h = 2160

def CRTXNet( n_classes ,  input_height, input_width ): # input_height = 360, input_width = 640

	imgs_input = Input(shape=(9,input_height,input_width))

	#layer1
	x = Conv2D(64, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first' )(imgs_input)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	#layer2
	x = Conv2D(64, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first' )(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	#layer3
	x = MaxPooling2D((2, 2), strides=(2, 2), data_format='channels_first' )(x)

	#layer4
	x = Conv2D(128, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first' )(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	#layer5
	x = Conv2D(128, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first' )(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	#layer6
	x = MaxPooling2D((2, 2), strides=(2, 2), data_format='channels_first' )(x)

	#layer7
	x = Conv2D(256, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first' )(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	#layer8
	x = Conv2D(256, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first' )(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	#layer9
	x = Conv2D(256, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first' )(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	#layer10
	x = MaxPooling2D((2, 2), strides=(2, 2), data_format='channels_first' )(x)

	#layer11
	x = ( Conv2D(512, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first'))(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	#layer12
	x = ( Conv2D(512, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first'))(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	#layer13
	x = ( Conv2D(512, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first'))(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	#layer14
	x = ( UpSampling2D( (2,2), data_format='channels_first'))(x)

	#layer15
	x = ( Conv2D( 256, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first'))(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	#layer16
	x = ( Conv2D( 256, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first'))(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	#layer17
	x = ( Conv2D( 256, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first'))(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	#layer18
	x = ( UpSampling2D( (2,2), data_format='channels_first'))(x)

	#layer19
	x = ( Conv2D( 128 , (3, 3), kernel_initializer='random_uniform', padding='same' , data_format='channels_first' ))(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	#layer20
	x = ( Conv2D( 128 , (3, 3), kernel_initializer='random_uniform', padding='same' , data_format='channels_first' ))(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	#layer21
	x = ( UpSampling2D( (2,2), data_format='channels_first'))(x)

	#layer22
	x = ( Conv2D( 64 , (3, 3), kernel_initializer='random_uniform', padding='same'  , data_format='channels_first' ))(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	#layer23
	x = ( Conv2D( 64 , (3, 3), kernel_initializer='random_uniform', padding='same'  , data_format='channels_first' ))(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	#layer24
	x =  Conv2D( n_classes , (3, 3) , kernel_initializer='random_uniform', padding='same', data_format='channels_first' )(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	o_shape = Model(imgs_input , x ).output_shape
	print ("layer24 output shape:", o_shape[1],o_shape[2],o_shape[3])
	#layer24 output shape: 256, 360, 640

	OutputHeight = o_shape[2]
	OutputWidth = o_shape[3]

	#reshape the size to (256, 360*640)
	x = (Reshape((  -1  , OutputHeight*OutputWidth   )))(x)

	#change dimension order to (360*640, 256)
	x = (Permute((2, 1)))(x)

	#layer25
	gaussian_output = (Activation('softmax'))(x)

	model = Model( imgs_input , gaussian_output)
	model.outputWidth = OutputWidth
	model.outputHeight = OutputHeight

	#show model's details
	model.summary()

	return model

def rename_files_to_numbers(dir_path):
    file_str = '0000'
    for path in os.listdir(dir_path):
        filename, ext = path.split('.')
        _, file_no = filename.split('_')
        file_path = dir_path + "\\" + path
        file_name = file_str[len(str(file_no)):] + str(file_no)
        new_file_path = dir_path + "\\" + file_name + "." + ext
        print (file_path, new_file_path)
        os.rename(file_path, new_file_path)

def create_label_file(home, label_file_name, data_dir):
    label_path = home + "\\" + label_file_name
    pics = glob.glob(home + "\\" + data_dir + "\\*.jpg")
    anno = glob.glob(home + "\\" + data_dir + "\\*.xml")
    label_header = ["FRAME NAME","Visibility Class","X","Y"]

    print ("***Start creating Label File***")
    with open(label_path, mode='w',newline='') as label_file:
        csv_writer = csv.writer(label_file, delimiter=',')
        csv_writer.writerow(label_header)
        for img_path in pics:
            img_info = []
            img_info.append(img_path.split("\\")[-1])
            filename = img_path.split("\\")[-1].split(".")[0]
            img_anno = img_path.split(".")[0] + ".xml"
            if (img_anno in anno):
                img_info.append(1)
                tree = ET.parse(img_anno)
                root = tree.getroot()
                bndbox = root.findall('./object/bndbox')[0]
                xmin = int(bndbox.find('xmin').text)
                ymin = int(bndbox.find('ymin').text)
                xmax = int(bndbox.find('xmax').text)
                ymax = int(bndbox.find('ymax').text)
                x = (xmin+xmax) // 2
                y = (ymin+ymax) // 2
                img_info.append(x)
                img_info.append(y)
            else:
                img_info.extend([0,0,0])
            csv_writer.writerow(img_info)
    print ("***Finished Creating Label File***")

def gaussian_kernel(variance):
    #create gussian heatmap 
    x, y = np.mgrid[-size:size+1, -size:size+1]
    g = np.exp(-(x**2+y**2)/float(2*variance))
    return g 

def generate_heatmap(home, data_dir, label_file_name, heat_map_dir):
    #make the Gaussian by calling the function
    gaussian_kernel_array = gaussian_kernel(variance)
    #rescale the value to 0-255
    gaussian_kernel_array =  gaussian_kernel_array * 255//gaussian_kernel_array[len(gaussian_kernel_array)//2][len(gaussian_kernel_array)//2]
    #change type as integer
    gaussian_kernel_array = gaussian_kernel_array.astype(int)

    #show heatmap 
    plt.imshow(gaussian_kernel_array, cmap=plt.get_cmap('gray'), interpolation='nearest')
    plt.colorbar()
    plt.show()

    #################change the path####################################################
    pics = glob.glob(home + "\\" + data_dir + "\\*.jpg")
    output_pics_path = home + "\\" + heat_map_dir + "\\" 
    label_path = home + "\\" + label_file_name
    ####################################################################################

    #check if the path need to be create
    if not os.path.exists(output_pics_path ):
        os.makedirs(output_pics_path)


    #read csv file
    with open(label_path, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        #skip the headers
        next(spamreader, None)  

        for row in spamreader:
                visibility = int(float(row[1]))
                FileName = row[0]
                #if visibility == 0, the heatmap is a black image
                if visibility == 0:
                    heatmap = Image.new("RGB", (img_w, img_h))
                    pix = heatmap.load()
                    for i in range(img_w):
                        for j in range(img_h):
                                pix[i,j] = (0,0,0)
                else:
                    x = int(float(row[2]))
                    y = int(float(row[3]))

                    #create a black image
                    heatmap = Image.new("RGB", (img_w, img_h))
                    pix = heatmap.load()
                    for i in range(img_w):
                        for j in range(img_h):
                                pix[i,j] = (0,0,0)

                    #copy the heatmap on it
                    for i in range(-size,size+1):
                        for j in range(-size,size+1):
                                if x+i<img_w and x+i>=0 and y+j<img_h and y+j>=0 :
                                    temp = gaussian_kernel_array[i+size][j+size]
                                    if temp > 0:
                                        pix[x+i,y+j] = (temp,temp,temp)
                #save image
                heatmap.save(output_pics_path + "/" + FileName.split('.')[-2] + ".jpg", "JPEG")

def generate_train_test_pairs(home, training_file_name, testing_file_name, data_dir, heat_map_dir, label_file_name):
    visibility_for_testing = []

    with open(training_file_name,'w') as file:
        for index in range(1,82):
            #################change the path####################################################
            #images_path = home + "\\cric_orig_data\\"
            #annos_path = home + "\\heat_map\\"
            images_path = home + "\\" + data_dir + "\\"
            print('images_path', images_path)
            annos_path = home + "\\" + heat_map_dir + "\\"

            ####################################################################################

            images = glob.glob( images_path + "*.jpg"  ) + glob.glob( images_path + "*.png"  ) +  glob.glob( images_path + "*.jpeg"  )
            #print(images)
            images.sort()
            annotations  = glob.glob( annos_path + "*.jpg"  ) + glob.glob( annos_path + "*.png"  ) +  glob.glob( annos_path + "*.jpeg"  )
            annotations.sort()
            
            #check if annotation counts equals to image counts
            assert len( images ) == len(annotations)
            for im , seg in zip(images,annotations):
    ##            import sys                 
    ##            print('result',im.split('/')); sys.exit(0)
                assert(  im.split('\\')[-1].split(".")[0] ==  seg.split('\\')[-1].split(".")[0] )

            visibility = {}
            with open(home + "\\" + label_file_name, 'r') as csvfile:
                spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
                #skip the headers
                next(spamreader, None)  
                
                for row in spamreader:
                    visibility[row[0]] = row[1]
                        
                        
            #output all of images path, 0000.jpg & 0001.jpg cant be used as input, so we have to start from 0002.jpg
            for i in range(2,len(images)): 
                    #remove image path, get image name   
                    #print(i);sys.exit(0)
                    file_name = images[i].split('\\')[-1]
                    #print('file_name', file_name)
    ##                #visibility 3 will not be used for training
    ##                if visibility[file_name] == '3': 
    ##                    visibility_for_testing.append(images[i])
                    #check if file image name same as annotation name
                    assert(  images[i].split('\\')[-1].split(".")[0] ==  annotations[i].split('\\')[-1].split(".")[0] )
                    #write all of images path
                    file.write(images[i] + "," + images[i-1] + "," + images[i-2] + "," + annotations[i] + "\n")
                    
    file.close()

    #read all of images path
    lines = open(training_file_name).read().splitlines()

    #70% for training, 30% for testing 
    training_images_number = int(len(lines)*0.7)
    testing_images_number = len(lines) - training_images_number
    print ("Total images:", len(lines), "Training images:", training_images_number,"Testing images:", testing_images_number)

    #shuffle the images
    random.shuffle(lines)
    #training images
    with open(training_file_name,'w') as training_file:
        training_file.write("img, img1, img2, ann\n")
        #testing images
        with open(testing_file_name,'w') as testing_file:
            testing_file.write("img, img1, img2, ann\n")
            
            #write img, img1, img2, ann to csv file
            for i in range(0,len(lines)):
                if lines[i] != "":
                    if training_images_number > 0 and lines[i].split(",")[0] not in visibility_for_testing :
                        training_file.write(lines[i] + "\n")
                        training_images_number -=1
                    else:
                        testing_file.write(lines[i] + "\n")
                        
    training_file.close()
    testing_file.close()

#get input array
def getInputArr( path ,path1 ,path2 , width , height):
    try:
        #read the image
        img = cv2.imread(path, 1)
        #resize it 
        img = cv2.resize(img, ( width , height ))
        #input must be float type
        img = img.astype(np.float32)

        #read the image
        img1 = cv2.imread(path1, 1)
        #resize it 
        img1 = cv2.resize(img1, ( width , height ))
        #input must be float type
        img1 = img1.astype(np.float32)

        #read the image
        img2 = cv2.imread(path2, 1)
        #resize it 
        img2 = cv2.resize(img2, ( width , height ))
        #input must be float type
        img2 = img2.astype(np.float32)

        #combine three imgs to  (width , height, rgb*3)
        imgs =  np.concatenate((img, img1, img2),axis=2)

        #since the odering of CRTXNet  is 'channels_first', so we need to change the axis
        imgs = np.rollaxis(imgs, 2, 0)
        return imgs

    except Exception as e:

        print (path , e)



#get output array
def getOutputArr( path , nClasses ,  width , height  ):

    seg_labels = np.zeros((  height , width  , nClasses ))
    try:
        img = cv2.imread(path, 1)
        img = cv2.resize(img, ( width , height ))
        img = img[:, : , 0]

        for c in range(nClasses):
            seg_labels[: , : , c ] = (img == c ).astype(int)

    except Exception as e:
        print (e)
        
    seg_labels = np.reshape(seg_labels, ( width*height , nClasses ))
    return seg_labels



#read input data and output data
def InputOutputGenerator( images_path,  batch_size,  n_classes , input_height , input_width , output_height , output_width ):

    #read csv file to 'zipped'
    columns = defaultdict(list)
    with open(images_path) as f:
        reader = csv.reader(f)
        #reader.next()#python 2
        next(reader)# python 3
        for row in reader:
            for (i,v) in enumerate(row):
                columns[i].append(v)
    zipped = itertools.cycle( zip(columns[0], columns[1], columns[2], columns[3]) )

    while True:
        Input = []
        Output = []
        #read input&output for each batch
        for _ in range( batch_size) :
                        #path, path1, path2 , anno = zipped.next()
                        path, path1, path2 , anno = next(zipped)
                        Input.append( getInputArr(path, path1, path2 , input_width , input_height))
                        Output.append( getOutputArr( anno , n_classes , output_width , output_height))
        #return Input&Output
        yield np.array(Input) , np.array(Output)


def train_model(training_images_name, train_batch_size, n_classes, input_height, input_width,
                save_weights_path, epochs, load_weights, step_per_epochs,
                optimizer_name):
    
    #load CRTXNet model
    m = CRTXNet( n_classes , input_height=input_height, input_width=input_width   )
    m.compile(loss='categorical_crossentropy', optimizer= optimizer_name, metrics=['accuracy'])

    #check if need to retrain the model weights
    if load_weights != "-1":
            m.load_weights("weights/model." + load_weights)

    #show CRTXNet details, save it as CRTXNet.png
    #plot_model( m , show_shapes=True , to_file='CRTXNet.png')

    #get CRTXNet output height and width
    model_output_height = m.outputHeight
    model_output_width = m.outputWidth

    #creat input data and output data
    Generator  = InputOutputGenerator( training_images_name,  train_batch_size,  n_classes , input_height , input_width , model_output_height , model_output_width)

    #start to train the model, and save weights until finish 
    #'''
    m.fit_generator( Generator, step_per_epochs, epochs )
    m.save_weights( save_weights_path + ".0" )

    #'''

    #start to train the model, and save weights per 50 epochs  

    ##for ep in range(1, epochs+1 ):
    ##	print "Epoch :", str(ep) + "/" + str(epochs)
    ##	m.fit_generator(Generator, step_per_epochs)
    ##	if ep % 50 == 0:
    ##		m.save_weights(save_weights_path + ".0")


if __name__== '__main__':
    #Renaming file names to numbers (0.jpg, 1.jpg)
    #dir_path = r'C:\Users\User\Desktop\ball\CRTXNET\Dataset\cric_orig_data'
    #rename_files_to_numbers(dir_path)

    #Creating the Label .CSV File
    home = r'C:\Users\User\Desktop\ball\CRTXNET\Dataset'
    label_file_name = "label.csv"
    data_dir = "cric_orig_data"
    #create_label_file(home, label_file_name, data_dir)

    #Generating the heatmaps as ground truth
    #create the heatmap as ground truth
    heat_map_dir = 'heat_map'
    #generate_heatmap(home, data_dir, label_file_name)

    #Generating Train and Test Pairs
    training_file_name = "training_model.csv"
    testing_file_name = "testing_model.csv"
    #generate_train_test_pairs(home, training_file_name, testing_file_name, data_dir, heat_map_dir, label_file_name)

    #parse parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_weights_path", type = str  )
    parser.add_argument("--training_images_name", type = str  )
    parser.add_argument("--n_classes", type=int )
    parser.add_argument("--input_height", type=int , default = 360  )
    parser.add_argument("--input_width", type=int , default = 640 )
    parser.add_argument("--epochs", type = int, default = 1000 )
    parser.add_argument("--batch_size", type = int, default = 2 )
    parser.add_argument("--load_weights", type = str , default = "-1")
    parser.add_argument("--step_per_epochs", type = int, default = 200 )

    args = parser.parse_args()
    training_images_name = args.training_images_name
    train_batch_size = args.batch_size
    n_classes = args.n_classes
    input_height = args.input_height
    input_width = args.input_width
    save_weights_path = args.save_weights_path
    epochs = args.epochs
    load_weights = args.load_weights
    step_per_epochs = args.step_per_epochs
    optimizer_name = optimizers.Adadelta(lr=1.0)

    #Train Full Model
    train_model(training_images_name, train_batch_size, n_classes, input_height, input_width,
                save_weights_path, epochs, load_weights, step_per_epochs,
                optimizer_name)

    ##For Testing Run the test.py with path to the test Video
