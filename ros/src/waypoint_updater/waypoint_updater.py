#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Int32
from scipy.spatial import KDTree
import math
import numpy as np

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 100 # Number of waypoints we will publish. You can change this number
DECEL = 1 / LOOKAHEAD_WPS 

class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb, queue_size=2)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb, queue_size=8)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
        rospy.Subscriber('/obstacle_waypoint', Lane, self.obstacle_cb)
        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below
        self.base_pose = None
        self.base_wp = None
        self.wp2d = None
        self.wpkdTree = None
        self.stopline_wp_idx = -1
        self.decelerate_count = 0
        # rospy.spin()

        self.loop()

    def loop(self):
        # code from Waypoint Updater partial Walkthrough lesson
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            if self.base_pose and self.base_wp:
                closest_wp_Idx=self.get_closest_wp_idx()
                self.publishwp(closest_wp_Idx)
            rate.sleep()

    def get_closest_wp_idx(self):
        # code from Waypoint Updater partial Walkthrough lesson
        x = self.base_pose.pose.position.x
        y = self.base_pose.pose.position.y
        idx = self.wpkdTree.query([x, y], 1)[1]

        clcoord = self.wp2d[idx]
        prevcoord = self.wp2d[idx-1]
        
        cl_vect = np.array(clcoord)
        prev_vect = np.array(prevcoord)
        pos_vect = np.array([x, y])

        val = np.dot(cl_vect -prev_vect, pos_vect-cl_vect)

        if val > 0:
            idx = (idx+1)%len(self.wp2d)
        
        return idx

    def publishwp(self, idx):
        # code from Waypoint Updater partial Walkthrough lesson
        lane = Lane()
        lane.header = self.base_wp.header
        if (self.stopline_wp_idx == -1) or (self.stopline_wp_idx >= idx + LOOKAHEAD_WPS):
            lane.waypoints = self.base_wp.waypoints[idx:idx + LOOKAHEAD_WPS]
        else:
            lane.waypoints = self.decelerate_waypoints(self.base_wp.waypoints[idx:idx + LOOKAHEAD_WPS], idx)
                
        self.final_waypoints_pub.publish(lane)
    def decelerate_waypoints(self, waypoints, closest_idx):
        temp = []
        for i, wp in enumerate(waypoints):
            p = Waypoint()
            p.pose = wp.pose

            # Distance includes a number of waypoints back so front of car stops at line
            stop_idx = max(self.stopline_wp_idx - closest_idx - 5, 0)
            dist = self.distance(waypoints, i, stop_idx)
            vel = math.sqrt(2*0.45*dist)
            if vel < 1.0:
                vel = 0.0
            p.twist.twist.linear.x = min(vel, wp.twist.twist.linear.x)
            temp.append(p)
            # rospy.logwarn("lane.waypoints {} ".format(len(lane.waypoints)))
        return temp

    def pose_cb(self, msg):
        self.base_pose = msg

    def waypoints_cb(self, waypoints):
        self.base_wp = waypoints
        if not self.wp2d:
            self.wp2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
            self.wpkdTree = KDTree(self.wp2d)

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        if self.stopline_wp_idx != msg.data:
            self.stopline_wp_idx = msg.data

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
