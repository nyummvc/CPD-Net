import torch
import LoaderFish
import os
import sys
os.environ['CUDA_DEVICE_ORDER']="PCI_BUS_ID"
import numpy as np
import matplotlib.pylab as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
import geotnf.point_tnf
import numpy as np
import sugartensor as tf
import os
import tqdm

######################################################
def write_to_tfrecords(dataMat,tfFileDirName):
    """
    example:
    write_to_tfrecords(dictionary{'x':np.array,...},'./Data/digits.tfrecords') Needs to remember the dimension and name of the dictionary data.
    return: tfrecords saved in given tfFileDirName
    """
    varNames=[i for i in list(dataMat.keys())]
    
    tmpData={}
    tmpShape={}
    
    for i in varNames:
        tmpData[i]=dataMat[i]
        tmpShape[i]=dataMat[i].shape
  
    ####Check shape and Nan############
    if (len(set([tmpShape[i][0] for i in tmpShape.keys()]))!=1) or \
    (np.sum([np.isnan(tmpData[i]).sum() for i in tmpData.keys()])!=0):
        print("Unbalance Label or NaN in Data")
        return
    ###################################
    writer = tf.python_io.TFRecordWriter(tfFileDirName)
    for i in range(len(tmpData[varNames[0]])):
        tmpFeature={}
        for ii in varNames:
            tmp=np.asarray(tmpData[ii][i], dtype=np.float32).tobytes()
            
            tmpFeature[ii]=tf.train.Feature(bytes_list=tf.train.BytesList(value=[tmp]))
            
        example = tf.train.Example(features=tf.train.Features(feature=tmpFeature))    
        writer.write(example.SerializeToString())
    writer.close()  
    print("writing successfully in your dir:{}".format(tfFileDirName))
    
    
def read_from_tfrecords(tfFileDirName,varNames,sizeBatch,shape,shuffle=True,rs=888):
    """
    example:
    read_from_tfrecords('./Data/digits.tfrecords',['x','y'],32,[[28,28],[1]])
    
    return: list of tensors. (this function should be only used in tensorflow codes)
    """
    varNames=list(varNames)
    tmp=[np.asarray(i,dtype=np.int32) for i in shape]
    shape=[]
    for i in tmp:
        if np.sum(np.shape(i))>1:
            shape.append(list(i))
        else:
            shape.append([int(i)])
    
    filename_queue = tf.train.string_input_producer([tfFileDirName])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    tmpFeatures={}
    for ii in varNames:
        tmpFeatures[ii]=tf.FixedLenFeature([], tf.string)
    tmpFeatures = tf.parse_single_example(serialized_example,
                                       features=tmpFeatures)  
    tmpVar=[]
    for i in range(len(varNames)):
        ii=varNames[i]
        tmp=tf.decode_raw(tmpFeatures[ii], tf.float32)
        tmp=tf.reshape(tmp, shape=list(shape[i]))
        tmpVar.append(tmp)
        
    if shuffle:
        tmpBatch=tf.train.shuffle_batch(tmpVar, sizeBatch, capacity=sizeBatch * 128,
                              min_after_dequeue=sizeBatch * 32, name=None, seed=rs)
    else:
        tmpBatch=tf.train.batch(tmpVar, sizeBatch, capacity=sizeBatch * 128, name=None)
        
    return tmpBatch    

def pairwise_dist(xt,y_p):
    a=xt.shape[1]
    b=y_p.shape[1]
    dist=tf.tile(tf.expand_dims(y_p,1),[1,a,1,1])-tf.tile(tf.expand_dims(xt,2),[1,1,b,1])
    dist=(dist[:,:,:,0]**2+dist[:,:,:,1]**2)
    return dist

def chamfer_loss(A,B):    
    r=tf.reduce_sum(A*A,2)
    r=tf.reshape(r,[int(r.shape[0]),int(r.shape[1]),1])
    r2=tf.reduce_sum(B*B,2)
    r2=tf.reshape(r2,[int(r.shape[0]),int(r.shape[1]),1])
    t=(r-2*tf.matmul(A, tf.transpose(B,perm=[0, 2, 1])) + tf.transpose(r2,perm=[0, 2, 1]))
    return tf.reduce_mean((tf.reduce_min(t, axis=1)+tf.reduce_min(t,axis=2))/2.0)

def Net(aa, yt, x):
    s=aa.shape[1]
    with tf.sg_context(name='NNReg', stride=1, act='leaky_relu', bn=True, reuse=tf.AUTO_REUSE): 
        yt=tf.expand_dims(yt,2)
        
        v1=tf.expand_dims(x,2).sg_conv(dim=16, size=(1,1),  name='gen9',pad="SAME",bn=True)        
        v2=v1.sg_conv(dim=64, size=(1,1),  name='gen1',pad="SAME",bn=True)        
        v3=v2.sg_conv(dim=128, size=(1,1),  name='gen2',pad="SAME",bn=True) 
        v4=v3.sg_conv(dim=256, size=(1,1),  name='gen3',pad="SAME",bn=True) 
        v5=v4.sg_conv(dim=512, size=(1,1),  name='gen4',pad="SAME",bn=True) 
        v5=tf.tile(tf.expand_dims(tf.reduce_max(v5, axis=1),axis=1),[1,s,1,1])
        vv5=v5
        
        v1=yt.sg_conv(dim=16, size=(1,1),  name='gen99',pad="SAME",bn=True)        
        v2=v1.sg_conv(dim=64, size=(1,1),  name='gen11',pad="SAME",bn=True)        
        v3=v2.sg_conv(dim=128, size=(1,1),  name='gen22',pad="SAME",bn=True) 
        v4=v3.sg_conv(dim=256, size=(1,1),  name='gen33',pad="SAME",bn=True) 
        v5=v4.sg_conv(dim=512, size=(1,1),  name='gen44',pad="SAME",bn=True) 
        v5=tf.tile(tf.expand_dims(tf.reduce_max(v5, axis=1),axis=1),[1,s,1,1])
        
        ff=tf.concat([tf.expand_dims(aa,2),v5], axis=-1) 
        ff=tf.concat([ff,vv5], axis=-1) 
        f1=ff.sg_conv(dim=256, size=(1,1),  name='f1',pad="SAME",bn=True)  
        f2=f1.sg_conv(dim=128, size=(1,1),  name='f2',pad="SAME",bn=True)  
        
        f3=f2.sg_conv(dim=2, size=(1,1),  name='f3',pad="SAME",bn=False, act="linear")  
        f3=tf.squeeze(f3,axis=2)
        
    return f3

def train():
    tf.Graph()
    tf.set_random_seed(888)
    print("*****************************************")
    print("Training started with random seed: {}".format(111))
    print("Batch started with random seed: {}".format(111))
    
    #read data
    x,y=read_from_tfrecords(tfname,
                                 ["source","target"], batSize, [[s1,2],[s2,2]])
    global_step = tf.Variable(1, trainable=False,name='global_step')
    yp=Net(x,x,y)+x    
    Loss=chamfer_loss(yp,y)    

    #Learning Rate****************************************************************************
    lr = tf.train.exponential_decay(learningRate, global_step,
                                                  batSize, learningRateDecay, staircase=False) 
    # Optimization Algo************************************************************************
    train_step = tf.train.AdamOptimizer(learning_rate=lr,
                                                    beta1=adam_beta1,
                                                    beta2=adam_beta2
                                                   ).minimize(Loss,global_step=global_step)
    
    saver = tf.train.Saver(max_to_keep=int(maxKeepWeights))
    init_op = tf.group(tf.global_variables_initializer(),
                               tf.local_variables_initializer())
    
    # Continue Training************************************************************************
    if len(conWeightPath)>0:
        print("Continue Training...")
        tmp_var_list={}
        if len(conWeightVar)==0:
            print("For all variables")
            globals()['conWeightVar']={''}
        else:
            print("Training variables: {}".format(conWeightVar))
            
        for j in conWeightVar: 
            for i in tf.global_variables():
                if i.name.startswith(j):
                    tmp_var_list[i.name[:-2]] = i      
        saver1=tf.train.Saver(tmp_var_list)     
    
    # Training**********************************************************************************    
    with tf.Session() as sess:
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        # Read Weight******************************
        if len(conWeightPath)>0:
            print(conWeightPath)
            if stepsContinue==-1:            
                STEPS=sorted([int(i.split("/")[-1].split(".")[1].split("-")[-1]) for i in glob.glob(conWeightPath+"/*meta")])
                print("hahaha",STEPS)
                globals()['stepsContinue']=STEPS[-1]
                
            wtt=glob.glob(conWeightPath+"/*{}*meta".format(stepsContinue))[0][:-5]
            print("Reading Weight:{}".format(wtt))
            saver1.restore(sess,wtt)
            print('Weight is successfully updated from: {}'.format(wtt))  
        #*******************************************    
        stepst = sess.run(global_step)
        for t in tqdm.tqdm(range(stepst,int(maxStep)+1)):      
            _= sess.run([train_step]) 
            if t % saveStep==0:
                if not os.path.exists(dirSave):
                    os.makedirs(dirSave)
                saver.save(sess, dirSave + '/model.ckpt', global_step=t)
        coord.request_stop()
        coord.join(threads)   
################################################################
os.environ['CUDA_VISIBLE_DEVICES']="6" #change it to the number of gpu. Only single gpu is requried. 
train_num=20000 # number of sythesized training pairs
#deformation_list=[0.9,1.2,1.5,2.0]
deformation_list=[0.4]
batSize=8
maxStep=100000 # fixed with learningRate and learningRateDecay
learningRate=0.001 
learningRateDecay=0.999
adam_beta1=0.9 # check adam optimization
adam_beta2=0.99
conWeightVar=['NNReg','global_step'] # variables to be loaded
saveStep=20000 # frequency to save weight
maxKeepWeights=2000 # how many records to save (for disk)
stepsContinue=-1  # from which steps continu.
#For Debug and results printing
keepProb=0.99999
# Totally dosen't work if put dropout after max-pool layer. 
printStep=1000
s1=91
s2=91
clas="fish"
##############################################################
for def_level in deformation_list:
# unmark it if you need to generate your own dataset for training. 
#     a=LoaderFish.PointRegDataset(total_data=train_num, 
#              deform_level=def_level,
#              noise_ratio=0, 
#              outlier_ratio=0, 
#              outlier_s=False,
#                outlier_t=False, 
#                noise_s=False, 
#                noise_t=False,
#              missing_points=0,
#              miss_source=False,
#                miss_targ=False,
#                clas=1)

#     write_to_tfrecords({"source":np.asanyarray([i.T for i in a.target_list])[np.random.choice(range(train_num),train_num)],
#                    "target":np.asanyarray([i.T for i in a.target_list])},"Def_train_{}_{}_{}.tfrecords".format(clas,def_level,train_num))

    tfname="Def_train_{}_{}_{}.tfrecords".format(clas,def_level,train_num)
    dat="Exp1"
    dirSave="./UnSup-{}/{}_Def_{}_trNum_{}_maxStep_{}".format(clas,
        dat, def_level,train_num, maxStep)
    conWeightPath=""
    train()
