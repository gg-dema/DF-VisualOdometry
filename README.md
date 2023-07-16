
# DFVO: 
Final exam for the Vision&Perception class, Sapienza, AI & Robotics, 2023

# The project
DFVO is an algorithm for a visual odometry system with monocular vision. The goal of the project is to establish a good trade-of between the classical technique for vision systems and NeuralNetwork-based systems. 

It's composed of 2 different trackers, the E-tracker, (based on the epipolar constraint), and a PnP-tracker (Perspective-n-Point).


## 


[Reference paper: DFVO-Visual Odometry Revisited: What Should Be Learnt?](https://arxiv.org/abs/1909.09803)
![image](https://github.com/gg-Dema/DF-VisualOdometry/assets/99049717/a5a434cb-0223-43ea-aafb-f6c7a764bad4)




## implementation and usage
All the networks used are pretrained on the kitti dataset.

For easy usage we create some interfaces to the [flow net](https://github.com/gg-Dema/DF-VisualOdometry/blob/main/flow_net_ui.py) and to the [depth net](https://github.com/gg-Dema/DF-VisualOdometry/blob/main/depth_net/depth_net.py) These networks are used for features selection and depth estimation.
We did not manually implement these networks, but used the cited implementations, with associated pretrained weights. 
To be clear, in the folder "flow_net" we only wrote the script "flow_net_ui.py", in the folder "depth_net" we only worte the script "depth_net.py". The remaining files are taken from the linked repositeries

The file [tracker_v4](https://github.com/gg-Dema/DF-VisualOdometry/blob/main/tracker_v4.py) contains another interface and the 2 tracker implementation
To choose between which tracker is more suitable for a particular frame, we calculate the GRIC score, a model selection metric based on the number of inliers between the epipolar and homography transformation

Our system is capable of computing the relative movement between consecutive frames and saving the trajectory of the agent. We are still facing problems with the amplitude of the curve.

In the main file it's possible to run all the experiments by command line using our parser
## Dataset and evaluation 
The main reference dataset is the Kitti dataset, a common benchmark for visual odometry systems in the field of autonomous driving. \
To evaluate the performance of the system, we also test our procedure on a standard drone dataset, where the top view allows us to evaluate easily the correctness of the system
## Authors

- [@Gabriele Di Marzo](https://www.github.com/gg-Dema)
- [@Andrea Ferrari](https://github.com/Andryf00)
