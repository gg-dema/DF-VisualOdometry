



from flow_net.flow_utils import Network, estimate
import math
import torch
import numpy as np
import PIL
import PIL.Image
import matplotlib.pyplot as plt

N_PATCHES = 10
N = 2000
THRESHOLD = -1000

class Flow_net_ui_older():
    def __init__(self) -> None:
        self.net = Network().to("cuda")
        self.net.eval()
                

    """
    Warp function, In input we get forward and backward flow, for each point y,x we follow the corresponding forward flox dy,dx 
    and find ourselves in p0_=y+dy, x+dx. Then we find the backward flow dby, dbx in p0_, ideally dy=-dby.
    we return for each x,y dbx, dby.
    """
    def warp(self, forward, backward):
        #remember that flow is a 2 layer matrix, with horizontal and vertical flow:
        fx = forward[0]
        fy = forward[1]
        warped = torch.FloatTensor(size=forward.size())

        max_y = forward.size()[1]
        max_x = forward.size()[2]
        for y in range(max_y):
            for x in range(max_x):
                #horizontal and vertical flows in x,y:
                dx=int(fx[y,x])
                dy=int(fy[y,x])
                try:
                    #follow the flow and find ourselves in p0_=y+dy, x+dx. p0_ is the match of y,x
                    p0_=tuple(map(sum, zip((y,x), (dy,dx))))
                    #sometimes the flow might lead to out of image points, we discard those
                    if p0_[0]>=max_y or p0_[1]>=max_x or p0_[0]<0 or p0_[1]<0: warped[:,y,x] = 10000
                    #backward flow in p0_
                    else: warped[:,y,x] = backward[:, p0_[0], p0_[1]]
                except Exception:
                    print(y,x,dx,dy,p0_) 
                    exit()
        return warped
    
    def get_matches(self, path_1, path_2, mode = 'local_top_k', draw = False):
        tenOne = torch.FloatTensor(np.ascontiguousarray(np.array(PIL.Image.open(path_1).resize((640, 192)))[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0)))
        tenTwo = torch.FloatTensor(np.ascontiguousarray(np.array(PIL.Image.open(path_2).resize((640, 192)))[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0)))

        #to measure flow consistency we need forward and backwards flow
        with torch.no_grad():
            forward = estimate(tenOne, tenTwo, self.net)
            backward = estimate(tenTwo, tenOne, self.net)

        w = self.warp(forward, backward)
        #consistency: in the paper it's actually -forwad -warped, but it doesn't work so whatever
        C = forward - w
        C_total = C[0] + C[1]
        rows = []
        cols = []
        patch_x = int(C_total.shape[1] / N_PATCHES)#1024
        patch_y = int(C_total.shape[0] / N_PATCHES)#436

        #N_PATCHES*N_PATCHES total patches
        if mode == 'local_top_k':
            for i in range(N_PATCHES):
                for j in range(N_PATCHES):
                    q = 0
                    #select current patch
                    partial_c = C_total[patch_y*i:patch_y*(i+1), patch_x*j:patch_x*(j+1)]
                    flat_partial = partial_c.flatten()
                    #check how many pixels have consistency above threshold, we will only select pixels that are above threshold
                    for p in flat_partial:
                        if p>THRESHOLD: q+=1 #q=nr of pixel with consistency above threshold
                    #esither select the top N/N_PATCHES in the best case, but if there are a lot of pixels with C lower than threshold select the remaining q
                    k=min(int(N/(N_PATCHES**2)), q)
                    top_v, top_i = torch.topk(input=flat_partial, k=k)

                    rows_p, cols_p = np.unravel_index(top_i, partial_c.shape)#unraveled relatively to the current patch
                    #unraveled relatively to the whole img
                    rows+=[r + patch_y*i for r in rows_p]
                    cols+=[c + patch_x*j for c in cols_p]

        #TOP N OVER THE WHOLE IMAGE, but top k locally is better
        if mode == 'top_n':
            flattened_c = C_total.flatten()
            top_v, top_i = torch.topk(input=flattened_c, k=1000)
            rows, cols = np.unravel_index(top_i, C_total.shape)

        
        fx = forward[0] #horizontal flow
        fy = forward[1] #vertical flow
        matched_rows=[]
        matched_cols=[]
        #iterate over all the points selected as matchings, reminder that (rows, cols) includes every point to be matched
        for i in range(len(rows)):
            p0=(rows[i],cols[i]) #point to be matched
            dx=int(fx[p0])
            dy=int(fy[p0])
            p0_=tuple(map(sum, zip(p0, (dy,dx)))) #matching point, obtained by following the flow in p0
            matched_rows.append(p0_[0]) 
            matched_cols.append(p0_[1])
        if draw:
            for i in range(len(rows)):
                p0=(rows[i],cols[i]) #point to be matched
                p0_=(matched_rows[i], matched_cols[i])
                #print(p0, "+", dy,dx,"=",p0_, w[:,p0[0],p0[1]])

                #color in red the matching pixels:
                tenOne[:,p0[0], p0[1]] = torch.FloatTensor([255,0,0])
                tenTwo[:,p0_[0], p0_[1]] = torch.FloatTensor([255,0,0])
            fig, axes = plt.subplots(1, 2)
            axes[0].imshow(tenOne.permute(1,2,0).detach().numpy())
            axes[1].imshow(tenTwo.permute(1,2,0).detach().numpy())
            plt.tight_layout()
            plt.show()
        return cols, rows, matched_cols, matched_rows
