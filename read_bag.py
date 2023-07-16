import cv2
import numpy as np
from pathlib import Path
from rosbags.highlevel import AnyReader


"""
Utils to load images from a rosbag, it is needed since the drone dataset is found in ros bags"""

STEP = 5 #how many frames to skip

with AnyReader([Path("data_drone/40_4.bag")]) as reader:
    connections = [x for x in reader.connections if x.topic == '/right/downsample_raw']
    i=0
    imgs=[]
    for connection, timestamp, rawdata in reader.messages(connections=connections):
        i+=1
        msg = reader.deserialize(rawdata, connection.msgtype)
        data = msg.data
        img = np.resize(data, ( msg.height, msg.width))
        imgs.append(img)
    video = np.array(imgs)
    for i in range(0, video.shape[0], STEP):

        img = np.stack(arrays=(video[i,:,:], video[i,:,:], video[i,:,:])).transpose(1,2,0)
        i = str(i).zfill(5)
        print(i)
        cv2.imwrite("data_drone/imgs_40_step_5/"+str(i) + ".jpg", img)  



