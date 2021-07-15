import PyKDL as kdl

import kdl_parser_py

from urdf_parser_py.urdf import URDF
import own_pykdl_utils
#from pykdl_utils.kdl_parser import kdl_tree_from_urdf_model

def main():
    robot = URDF.from_xml_file("/home/robot/catkin_ws/src/aubo_description/urdf/aubo_i3.urdf")
    tree = None
    tree = own_pykdl_utils.kdl_parser.kdl_tree_from_urdf_model(robot)
    chain = tree.getChain('base_link', 'wrist3_Link')
    print (chain.getNrOfJoints())
    kdl_kin = own_pykdl_utils.kdl_kinematics.KDLKinematics(robot, 'base_link', 'wrist3_Link')
    q = kdl_kin.random_joint_angles()
    pose = kdl_kin.forward(q) # forward kinematics (returns homogeneous 4x4 numpy.mat)
    q_ik = kdl_kin.inverse(pose, q+0.3) # inverse kinematics

    if q_ik is not None:
        pose_sol = kdl_kin.forward(q_ik) # should equal pose
    print(q)
    print(pose)
    print(q_ik)

if __name__ == '__main__':
    main()