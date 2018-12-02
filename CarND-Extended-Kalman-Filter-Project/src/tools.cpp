#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
	
  	VectorXd rmse(4);
	rmse << 0,0,0,0;

    // TODO: YOUR CODE HERE

	// check the validity of the following inputs:
	//  * the estimation vector size should not be zero
	//  * the estimation vector size should equal ground truth vector size
	// ... your code here
	if(estimations.size() != ground_truth.size() || estimations.size() == 0)
	{
	    cout << "Invalid estimation or ground_truth data" << endl;
	    return rmse;
	}
    unsigned int i = 0;
	//accumulate squared residuals
	for(; i < estimations.size(); ++i){
        // ... your code here
		VectorXd residuals = estimations[i] - ground_truth[i];
		residuals = residuals.array() * residuals.array();
		rmse += residuals;
	}
	//cout << "i:" << i << endl;
	//cout << "estimations.size():" << estimations.size() << endl;

	//calculate the mean
	// ... your code here
	rmse = rmse / estimations.size();

	//calculate the squared root
	// ... your code here
	rmse = rmse.array().sqrt();

	//return the result
	return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  	/**
  	TODO:
  	  * Calculate a Jacobian here.
  	*/
	MatrixXd Hj(3,4);
	//recover state parameters
	float px = x_state(0);
	float py = x_state(1);
	float vx = x_state(2);
	float vy = x_state(3);

	//TODO: YOUR CODE HERE 

	//check division by zero
	float lu_2 = px * px + py * py;

	if(fabs(lu_2) < 0.0001)
	{
	    cout << "Error - Division by zero" << endl;
	}
	else
	//compute the Jacobian matrix
	{
	    //float lu_2 = px * px + py * py;
	    float lu_1 = sqrt(lu_2);
	    float lu_3 = lu_1 * lu_1 * lu_1;
	    Hj << px / lu_1 , py / lu_1 , 0 , 0 ,
	          -py / lu_2 , px / lu_2 , 0 , 0 ,
	          py * (vx * py - vy * px) / lu_3 , px * (vy * px - vx * py) / lu_3 , px / lu_1 , py / lu_1;
	}

	return Hj;
}
