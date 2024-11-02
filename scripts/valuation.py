#!/usr/bin/env python3
import os
import sys
import argparse

import rospy
import roslib
roslib.load_manifest('ptp_make_dataset')
import numpy as np
from ptp_msgs.msg import PedestrianArray

class MakeDataset():
    def __init__(self):
        rospy.init_node('make_dataset_node', anonymous=True)
        self.curr_ped_sub = rospy.Subscriber(
            '/curr_ped',
            PedestrianArray,
            self.curr_ped_callback)

        if not rospy.has_param('~tag'):
            rospy.set_param('~tag', 'hoge')
        self.tag = rospy.get_param('~tag')

        self.pkg_path = roslib.packages.get_pkg_dir('ptp_make_dataset')

        self.dataset_dir = self.pkg_path + '/datasets/' 

        if not os.path.exists(self.dataset_dir):
            os.makedirs(self.dataset_dir)
        
    def curr_ped_callback(self, msg):
        data_array = np.array(msg.data, dtype=np.dtype(msg.dtype)).reshape(msg.shape)
        self.add_data(data_array)

    def add_data(self, data_array_):
        with open(self.dataset_dir + self.tag + '.txt', 'a') as f:
            for data in data_array_:
                f.write(f'{data[0]}\t{data[1]}\t{data[2]}\t{data[3]}\n')
                # print(f'{data[0]}\t{data[1]}\t{data[2]}\t{data[3]}\n')
            try:
                print(f'Written! frame: {data[0]}')
            except:
                print('No pedestrians in this frame')

def main():
    node = MakeDataset()
    rospy.spin()

if __name__ == '__main__':
    main()
