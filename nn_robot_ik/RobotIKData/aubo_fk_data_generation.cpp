#include <kdl/kdl.hpp> 
#include <kdl/chain.hpp> 
#include <kdl/tree.hpp> 
#include <kdl/segment.hpp> 
#include <kdl/chainfksolver.hpp> 
#include <kdl_parser/kdl_parser.hpp> 
#include <kdl/chainfksolverpos_recursive.hpp> 
#include <kdl/chainjnttojacsolver.hpp>
#include <kdl/frames_io.hpp> 
#include <kdl/utilities/error.h>
#include <kdl/chainiksolverpos_nr.hpp>
#include <kdl/chainiksolverpos_lma.hpp>
#include <kdl/chainiksolver.hpp>
#include <kdl/chainfksolver.hpp>
#include <kdl/frames.hpp>
#include "kdl_conversions/kdl_msg.h"
// #include "tf/transform_datatypes.h"
// #include "tf_conversions/tf_kdl.h"

#include <stdio.h> 
#include <iostream> 
#include <queue>
#include <sys/times.h>
#include <unistd.h>
#include <stdlib.h>
#include <std_msgs/Float64.h>
#include <cmath>
#include <vector>

using namespace KDL; 
using namespace std; 


JntArray _current_position(NO_OF_JOINTS);
JntArray current_velocity(NO_OF_JOINTS);
JntArray robot_position(NO_OF_JOINTS);
JntArray robot_velocity(NO_OF_JOINTS);
JntArray target_joint_pose(NO_OF_JOINTS);

const char* j_name_list[]={
"shoulder_joint",
"upperArm_joint",
"foreArm_joint",
"wrist1_joint",
"wrist2_joint",
"wrist3_joint",
};



int main(int argc,char** argv){
	Tree my_tree;
	kdl_parser::treeFromFile("/home/yan/catkin_ws/src/aubo_description/urdf/aubo_i3_soft_restrict.urdf", my_tree);

	Chain chain;
	my_tree.getChain("world","tcp_Link",chain);

	ChainFkSolverPos_recursive fksolver = ChainFkSolverPos_recursive(chain);
	unsigned int nj = chain.getNrOfJoints();
	unsigned int ns = chain.getNrOfSegments();
	printf("kinematics information: nj=%d, ns=%d\n",nj,ns);
	
	JntArray jointpositions = JntArray(nj);

	Frame cartpos;
	//used for time statistic
	double j1 = -180.0;
	double j2 = -180.0;
	double j3 = -180.0;
	double j4 = -180.0;
	double j5 = -180.0;
	double j6 = -180.0;

	while (j6 < 180){
		j1 += 1.0;
		if (j1 == 180){
			j1 = -180;
			j2 += 1.0;
		}
		if (j2 == 180){
			j2 = -180;
			j3 += 1.0;
		}
		if (j3 == 180){
			j3 = -180;
			j4 += 1.0;
		}
		if (j4 == 180){
			j4 = -180;
			j5 += 1.0;
		}
		if (j5 == 180){
			j5 = -180;
			j6 += 1.0;
		}


	}


	while(ros::ok()){


		fksolver.JntToCart(_current_position, eeFrame);


		vec[0] = _delta_pose.translation.x + eeFrame.p[0];
		vec[1] = _delta_pose.translation.y + eeFrame.p[1];
		vec[2] = _delta_pose.translation.z + eeFrame.p[2];
		

		Frame TargetFrame(rot, vec);
	}

}
