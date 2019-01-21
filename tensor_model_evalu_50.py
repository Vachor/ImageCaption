from keras.layers import Input,Embedding,Dropout,LSTM,Dense
from keras.layers.merge import add
from keras.models import Model
from keras.utils import plot_model
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.callbacks import ModelCheckpoint
from keras.initializers import Constant
from keras.models import load_model,model_from_yaml
import tensorflow as tf
import pickle
from scipy.misc import imread,imresize
import os
import copy
import numpy as np
import keras
from keras.objectives import categorical_crossentropy
from keras import backend as K
import p5_evaluate_tensor
import time
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(1)


def embedding_layer(max_voc_plus_1,tokenizer,max_sen_len):
    # comes from https://github.com/keras-team/keras/blob/master/examples/pretrained_word_embeddings.py
    word_index = tokenizer.word_index
    embedding_index = {}

    #make a embedding matrix
    with open(os.path.join('glove.6B','glove.6B.50d.txt'),'r',encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:],dtype=np.float32)
            embedding_index[word] = coefs
    embedding_matrix = np.zeros((max_voc_plus_1,50))

    for word,i in word_index.items():
        if word == 'startsen':
            embedding_matrix[i] = embedding_index.get('said')
        embedding_vector = embedding_index.get(word)
        if word == 'endsen':
            embedding_matrix[i] = embedding_index.get('percent')
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector



    #make a pretrain embedding layer
    #embedding_l = Embedding(max_voc_plus_1,50,embeddings_initializer=Constant(embedding_matrix),input_length=max_sen_len,trainable=False)
    return embedding_matrix

def define_Caption_Model(batchsize,max_voc_plus_1,max_sen_len,tokenizer):
    #input1 = Input(shape=(max_sen_len,))
    with tf.variable_scope('INPUT'):
        input1 = tf.placeholder(np.int32,shape=[None,max_sen_len])
        input2 = tf.placeholder(np.float32, shape=[None, 2048])
    # compare 2, different embedding layer
    with tf.variable_scope('RNN'):
        #embedding = Embedding(max_voc_plus_1,256,mask_zero=True)(input1)
        #embedding_parm = tf.get_variable('embedding',shape=[max_voc_plus_1,50],initializer=tf.glorot_uniform_initializer())
        embedding_parm = tf.constant(embedding_layer(max_voc_plus_1,tokenizer,max_sen_len),dtype=np.float32)
        word_embedding = tf.nn.embedding_lookup(embedding_parm,input1)
        time_steps = max_sen_len
        lstm = tf.contrib.rnn.BasicLSTMCell(50)
        initial_state = lstm.zero_state(batchsize, dtype=np.float32)
        out, current_state = lstm(word_embedding[:, 0], initial_state)
        out = ''
        for i in range(1,time_steps):
    # embedding_l = embedding_layer(max_voc_plus_1,tokenizer,max_sen_len)
    # embedding = embedding_l(input1)

    # with tf.variable_scope('word_embedding'):
    #     embedding = tf.Variable(np.identity(256, dtype=np.float32))
    #     weight = tf.nn.embedding_lookup()

    #dropout1 = Dropout(0.4)(embedding)
            out, current_state = lstm(word_embedding[:,i], current_state)
            tf.get_variable_scope().reuse_variables()
        final_out = out
        #lstm = LSTM(256)(dropout1)
    #input2 = Input(shape=(2048,))

    dropout2 = Dropout(0.4)(input2)
    #Dense1 = Dense(256,activation='relu')(dropout2)
    with tf.variable_scope('Dense1'):
        weight = tf.get_variable(name='weight',shape=[2048,50],initializer=tf.glorot_uniform_initializer())
        bias = tf.get_variable(name='bias',initializer=tf.random_normal([50]))
        dense1 = tf.matmul(dropout2,weight) + bias
        dense1 = tf.nn.relu(dense1)
    addl_D = final_out + dense1
    with tf.variable_scope('Dense2'):
        weight = tf.get_variable(name='weight', shape=[50, 1024], initializer=tf.glorot_uniform_initializer())
        bias = tf.get_variable(name='bias',initializer=tf.random_normal([1024]))
        dense2 = tf.matmul(addl_D, weight) + bias
        dense2 = tf.nn.relu(dense2)
    #dense2 = Dense(1024,activation='relu')(addl_D)
    with tf.variable_scope('Dense3'):
        #dense3 = Dense(max_voc_plus_1)(dense2)
        weight = tf.get_variable(name='weight',shape=[1024,max_voc_plus_1],initializer=tf.glorot_uniform_initializer())
        bias = tf.get_variable(name='bias',initializer=tf.random_normal([max_voc_plus_1]))
        dense3 = tf.matmul(dense2,weight) + bias
    #dense3 = keras.activations.softmax(dense3)
    return dense3,input1,input2,dense2,addl_D,word_embedding,lstm,dense1,initial_state,current_state

    #model = Model(inputs=[input1,input2],outputs=dense3)
    #plot_model(model,to_file='model.png',show_shapes=True)
    #adam = keras.optimizers.adam()
    #loss = keras.losses.categorical_crossentropy
    #model.compile(loss=loss,optimizer=adam)
    #return model

def train(batchsize,trainx,trainIMG,trainy,testx,testIMG,testy,learningrate,epoch,max_voc_plus_1,max_sen_len,tokenizer):
    pred,input1,input2,weight,addl_D,embedding,lstm,dense1,initial_state,final_state = define_Caption_Model(batchsize,max_voc_plus_1,max_sen_len,tokenizer)
    y = tf.placeholder(np.float32,shape=[None,7633])
    #loss = tf.reduce_mean(categorical_crossentropy(y,pred))
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learningrate).minimize(loss)
    saver = tf.train.Saver(max_to_keep=1)

    index = (np.arange(len(trainx)).astype(int))
    np.random.shuffle(index)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    min_val_loss = 9999
    val_total = 0
    count = 0
    counter = 0


    #for test
    tf.get_variable_scope().reuse_variables()
    dense3, input_1_te, input_2_te, _, _, _, _, _,_,_ = define_Caption_Model(1, max_voc_plus_1, max_sen_len, tokenizer)

    #for inheriting from the last batch state
    for i in range(epoch):

        #there is no input of state, so the initial state is 0
        next_state = sess.run(initial_state, feed_dict={input1: trainx[0:batchsize],
                                                        input2: trainIMG[0:batchsize]})
        next_state_for_val = sess.run(initial_state, feed_dict={input1: trainx[0:batchsize],
                                                        input2: trainIMG[0:batchsize]})

        with open('grammer.txt', 'w') as f:
            for start,end in zip(range(0,len(trainx),batchsize),range(batchsize,len(trainx),batchsize)):

                #print(sess.run(pred,feed_dict={input1:trainx[0:1],input2:trainIMG[0:1],y:trainy[0:1]}))

                _,loss_value,next_state,inistate = sess.run([optimizer,loss,final_state,initial_state],feed_dict={input1:trainx[start:end],input2:trainIMG[start:end],y:trainy[start:end],initial_state:next_state})
                print("Current Cost: ", loss_value, "\t Epoch {}/{}".format(i, epoch),
                        "\t Iter {}/{}".format(start, len(trainx)))


                # if counter % 100 == 0:
                #     img_features = get_IMG_features('example_bic.jpg')
                #     opt = ''
                #     step = 0
                #     caption = ['startsen']
                #     str = ' '.join(caption)
                #     input__1 = tokenizer.texts_to_sequences([str])
                #     input__1 = pad_sequences(input__1,maxlen=38)
                #
                #     while step<38 and opt != 'endsen':
                #         opt = sess.run(dense3, feed_dict={input_1_te: input__1, input_2_te: np.array([img_features], dtype=np.float32),
                #                                   y: trainy[0:1]})
                #         opt = np.argmax(opt)
                #         opt = idx_to_word(opt,tokenizer)
                #         caption.append(opt)
                #         print(caption)
                #         if step == 37 or opt == 'endsen':
                #             f.write(' '.join(caption)+'\n')
                #         str = ' '.join(caption)
                #         input__1 = tokenizer.texts_to_sequences([str])
                #         input__1 = pad_sequences(input__1, maxlen=38, dtype='float32')
                #         step += 1
                # counter = counter + 1

        # involve validation to get the optim model
        for start,end in zip(range(0,len(testx),batchsize),range(batchsize,len(testx),batchsize)):
            val_loss,next_state_for_val = sess.run([loss,final_state],feed_dict={input1:testx[start:end],input2:testIMG[start:end],y:testy[start:end],initial_state:next_state_for_val})
            val_total += val_loss
            count += 1
        val_loss_avg = val_total / count
        count = 0
        print('val_loss',val_loss_avg)
        if(val_loss_avg<min_val_loss):
            min_val_loss = val_loss_avg
            print('the loss of validation:',val_loss_avg)
            print('save model...')
        saver.save(sess,'RNNModel/mnist.ckpt',global_step=i+1)
        val_total = 0
def idx_to_word(optword,tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == optword:
			optword = word
			return word
def get_IMG_features(img):
	cnnmodel = load_model(os.path.join('CNNMODELS','ResNet50.h5'))
	img = imread(img)
	img = imresize(img,[224,224,3])
	features = cnnmodel.predict(np.asarray([img]))
	return features[0]
def get_max_sen_len(doc):
    alltext = []
    for key,value in doc.items():
        for i in value:
            alltext.append(i)

    max_sen_len = max(len(alltext[i].split()) for i,value in enumerate(alltext))

    return max_sen_len

def get_input_(filename,IMGFeatures,doc,tokenizer,max_sen_len,max_voc_plus_1):

    #put all image id (title) into variable list _title
    with open(filename,'r') as f:
        _title = []
        line = f.readline()
        _title.append(line.split('.')[0])
        while line:
            line = f.readline()
            _title.append(line.split('.')[0])   #this code may lead to a extra line append to the last line


    X = []
    Y = []
    XIMG = []
    for index in _title:

        #if reach to the last line, then continue the loop
        if len(index) <2:
            continue

        #make every caption into a list of number,and make output to be a one-shot code
        for des in doc[index]:
            des = tokenizer.texts_to_sequences([des])[0]
            for sta in range(1,len(des)):
                start = des[:sta]
                end = des[sta]
                X.append(pad_sequences([start],max_sen_len)[0])
                XIMG.append(IMGFeatures[index])
                Y.append(to_categorical(end,max_voc_plus_1))   #there is one more category, cause the 0 encode nothing about word
    return X,XIMG,Y

def main():


    #load some files from disk
    with open('ResNet50_c_fet.pkl', 'rb') as f:
        IMGFeatures = pickle.load(f)

    with open('Tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    with open('doc.pkl', 'rb') as f:
        doc = pickle.load(f)
    max_voc_plus_1 = len(tokenizer.word_index) + 1   #plus_one is for embedding layers
    max_sen_len = get_max_sen_len(doc)  #get the maximam length of sentences in all captions


    #define a deep learning model
    #model = define_Caption_Model(max_voc_plus_1, max_sen_len,tokenizer)
    #model_yaml = model.to_yaml()
    #with open(os.path.join('MODELS','archi.txt'),'w') as f:
    #	f.write(model_yaml)
    #get input of train dataset
    filename = os.path.join('Flickr8k_text', 'Flickr_8k.trainImages.txt')
    trainX, trainXIMG, trainY = get_input_(filename, IMGFeatures,doc,tokenizer,max_sen_len,max_voc_plus_1)
    trainX = np.asarray(trainX,dtype=np.float32)
    b = trainX[0:10]
    trainXIMG = np.asarray(trainXIMG)
    bb = trainXIMG[0:10]
    trainY = np.asarray(trainY)

    #get input of test dataset
    filename = os.path.join('Flickr8k_text', 'Flickr_8k.testImages.txt')
    testX, testXIMG, testY = get_input_(filename, IMGFeatures,doc,tokenizer,max_sen_len,max_voc_plus_1)
    testX = np.asarray(testX)
    testXIMG = np.asarray(testXIMG)
    testY = np.asarray(testY)

    #train model
    #filepath_model = os.path.join('MODELS','ep{epoch:03d}-val_loss{val_loss:.3f}.h5')
    # filepath_weight = os.path.join('MODELS','weight{epoch:03d}-val_loss{val_loss:.3f}.h5')
    # callbacks2 = ModelCheckpoint(filepath_weight,monitor='val_loss',save_best_only=True,save_weights_only=True, mode='min')
    # callbacks1 = ModelCheckpoint(filepath_model,monitor='valposs',save_best_only=True, mode='min')
    # model.fit([trainX, trainXIMG], trainY,batch_size=128, epochs=8, validation_data=([testX, testXIMG], testY),callbacks=[callbacks1,callbacks2])

    totalc = 6
    bleu1 = 0
    bleu2 = 0
    bleu3 = 0
    bleu4 = 0
    for i in range(totalc):

        train(128,trainX,trainXIMG,trainY,testX,testXIMG,testY,0.001,3,max_voc_plus_1,max_sen_len,tokenizer)
        tf.reset_default_graph()
        bleu1_,bleu2_,bleu3_,bleu4_ = p5_evaluate_tensor.main()
        tf.reset_default_graph()

        bleu1 += bleu1_
        print(bleu1)
        bleu2 += bleu2_
        print(bleu2)
        bleu3 += bleu3_
        print(bleu3)
        bleu4 += bleu4_
        print(bleu4)

    bleu1 /= totalc
    bleu2 /= totalc
    bleu3 /= totalc
    bleu4 /= totalc
    print(bleu1)
    print(bleu2)
    print(bleu3)
    print(bleu4)

if __name__ == '__main__':
    main()

