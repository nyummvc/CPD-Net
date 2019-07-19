import LoaderFish
import os
import sys
sys.path.append("../")
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
import glob
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

def chamfer_loss_np(A,B):    
    r=np.sum(A*A,2)
    r=np.reshape(r,[int(r.shape[0]),int(r.shape[1]),1])
    r2=np.sum(B*B,2)
    r2=np.reshape(r2,[int(r.shape[0]),int(r.shape[1]),1])
    t=(r-2*np.matmul(A, np.transpose(B,(0, 2, 1))) 
                                         + np.transpose(r2,(0, 2, 1)))
    return np.mean((np.min(t, axis=1)+np.min(t,axis=2))/2.0,axis=-1)

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

def test(tfname,weightPaths,steps=100000, Var=["NNReg"], lll=2000):
    tf.Graph()
    x,y=read_from_tfrecords(tfname,["source","target"], 10, [[91,2],[91,2]])
    global_step = tf.Variable(1, trainable=False,name='global_step')
    yp=Net(x,x,y)+x
    tmp_var_list={}
    for j in Var:
        for i in tf.global_variables():
            if i.name.startswith(j):
                tmp_var_list[i.name[:-2]] = i
                
    saver=tf.train.Saver(tmp_var_list)
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    path=weightPaths+"model.ckpt-{}".format(steps)
    
    Sour=[]
    Targ=[]
    Trans_S=[]
    
    with tf.Session() as sess:
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        saver.restore(sess,path)
        for i in tqdm.tqdm(range(lll)):
            S,T,TS=sess.run([x,y,yp])
            Sour.append(S)
            Targ.append(T)
            Trans_S.append(TS)
            
        coord.request_stop()
        coord.join(threads) 
    
    return Sour,Targ,Trans_S


def vis_Bat(xx,yy,yyp,name):
    fig = plt.figure(1, figsize=(20, 40))
    for i in range(8):
        x=xx[i]
        y=yy[i]
        yp=yyp[i]
        
        ax = fig.add_subplot(8, 4, i*4+1)
        plt.scatter(x[:,0],x[:,1], label="source",s=5,c="r")
        plt.scatter(y[:,0],y[:,1], label="target",s=20,c="b",marker='x')
        plt.ylim(-3.5, 3.5)
        plt.xlim(-3.5, 3.5)
        ax.axis('off')
       
        ax = fig.add_subplot(8, 4, i*4+2)
        plt.scatter(x[:,0],x[:,1], label="source",s=5,c="r")
        plt.scatter(yp[:,0],yp[:,1], label="transformed",s=5,c="r")
        for ii in range(len(x)):
            plt.arrow(x[ii,0],x[ii,1],(yp[ii,0]-x[ii,0]),(yp[ii,1]-x[ii,1]), head_width=0.03, head_length=0.08, fc='k', ec='k')
        
        plt.ylim(-3.5, 3.5)
        plt.xlim(-3.5, 3.5)
        ax.axis('off')

        ax = fig.add_subplot(8, 4, i*4+3)
        plt.scatter(y[:,0],y[:,1], label="target",s=20, c="b",marker="x")
        plt.scatter(yp[:,0],yp[:,1], label="transformed",s=5,c="r")
        plt.ylim(-3.5, 3.5)
        plt.xlim(-3.5, 3.5)
        ax.axis('off')
        
        ax = fig.add_subplot(8, 4, i*4+4)
        plt.scatter(x[:20,0],x[:20,1], label="source",s=5,c="r")
        plt.scatter(yp[:20,0],yp[:20,1], label="transformed",s=5,c="r")
        for ii in range(20):
            plt.arrow(x[ii,0],x[ii,1],(yp[ii,0]-x[ii,0]),(yp[ii,1]-x[ii,1]), head_width=0.04, head_length=0.04, fc='k', ec='k')
        
#         plt.ylim(-3.5, 3.5)
#         plt.xlim(-3.5, 3.5)
        ax.axis('off')
        
#     plt.show()
    plt.savefig(name,transparent=True)
    plt.close('all')
    
    
os.environ['CUDA_VISIBLE_DEVICES']="1"

DaTf=sorted(glob.glob("./Def_train_*.*_20000.tfrecords"))[:8]
Weig=sorted(glob.glob("./UnSup-fish/*"))

print("Data for training: ",DaTf)
print("Weight is loaded from: ", Weig)

Def_lv=[]
Bef_tr=[]
Aft_tr=[]
Bef_te=[]
Aft_te=[]

for i in range(len(Weig)):
    
    def_level=float(Weig[i].split("Def_")[-1].split("_")[0])
    print("deformation level : ", def_level)
    tr_d=DaTf[i]
    wt=Weig[i]

    Def_lv.append(def_level)
    
    test_num=200
    a=LoaderFish.PointRegDataset(total_data=test_num, 
                  deform_level=def_level,
                  noise_ratio=0, 
                  outlier_ratio=0, 
                  outlier_s=False,
                    outlier_t=False, 
                    noise_s=False, 
                    noise_t=False,
                  missing_points=0,
                  miss_source=False,
                    miss_targ=False)

    try:
        os.remove("temp_test_1.tfrecords")
    except:
        print("fine, you don't have such files")
    write_to_tfrecords({"source":np.asanyarray([i.T for i in a.target_list])[np.random.choice(range(test_num),test_num)],
                        "target":np.asanyarray([i.T for i in a.target_list])},"temp_test_1.tfrecords")

    S,T,TS=test("temp_test_1.tfrecords",wt+"/", lll=2)
    S=np.asanyarray(S).reshape(-1,91,2)
    T=np.asanyarray(T).reshape(-1,91,2)
    TS=np.asanyarray(TS).reshape(-1,91,2)
    org=chamfer_loss_np(T,S)
    aft=chamfer_loss_np(T,TS)
    Bef_te.append([np.mean(org), np.std(org)])
    Aft_te.append([np.mean(aft),np.std(aft)])
    if not os.path.exists("result_visuliation"):
        os.makedirs("result_visuliation")
    vis_Bat(S,T,TS,"result_visuliation/test.png")
    print("#####################################################")
    
print("C.D. for Inputs (mean+-std): ", Bef_te)
print("C.D. for Outputs (mean+-std): ", Aft_te)