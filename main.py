import argparse
from visual_odometry import VisualOdometry
from utils import *

def main(args):

    if args.test:
        vo_ui = VisualOdometry(args)
        print("Computing trajectory")
        vo_ui.compute_trajectory()
        print("Trajectory computed, check \"preds\output.txt\" for the predictions")
    else:
        print("Plotting trajectory")
        if args.dataset == 'drone':
            ground_truth = load_poses_from_txt_gt_drone("ground_truth_drone/40_4.txt")
            preds = load_poses_from_txt(args.input)
            plot_trajectory_drone(ground_truth, preds, args.output)
        else:
            ground_truth = load_poses_from_txt("data_odometry_poses/dataset/poses/02.txt")
            preds = load_poses_from_txt(args.input)
            plot_trajectory(ground_truth, preds, args.output)
        print("Trajectory plotted, check \"plots\output.pdf\" for the plot")
         

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Command-line argument parser example')

    parser.add_argument('-t', '--test', action='store_true', help='Compute a new trajectory if True, else evaluate an already computed trajectory')
    parser.add_argument('-i', '--input', help='Path to the input file, either a folder of images if -t, or a preds text file if not -t')
    parser.add_argument('-o', '--output', help='Name of the output file, it will be saved in either /plots or /preds depending on the flag --test')
    parser.add_argument('-d', '--dataset', help='Accepted values: {\'drone\', \'kitti\'}')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose mode')

    args = parser.parse_args()
    main(args)



