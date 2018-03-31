/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <random>

#include "particle_filter.h"

using namespace std;

//Creating a random number generator
default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	num_particles = 10;
	double weight_init = 1.0;

	//Creating normal distributions for x, y and theta for their initialization.
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	//Initializing all the particles.
	for(int i=0; i<num_particles; i++)
	{
		double x_ = dist_x(gen);
		double y_ = dist_y(gen);
		double theta_ = dist_theta(gen);

		Particle particle(i, x_, y_, theta_, weight_init);
		weights.push_back(weight_init);
		particles.push_back(particle);
	}
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	normal_distribution<double> dist_x(0.0, std_pos[0]);
	normal_distribution<double> dist_y(0.0, std_pos[1]);
	normal_distribution<double> dist_theta(0.0, std_pos[2]);

	vector<double> change(3);

	for(int i=0; i<particles.size(); i++)
	{
		Particle particle = particles[i];
		if(yaw_rate < 0.001)	
		{
			change[0] = velocity * cos(particle.theta) * delta_t;
			change[1] = velocity * sin(particle.theta) * delta_t;
		}
		else
		{
			change[0] = (velocity/yaw_rate) * (sin(particle.theta + yaw_rate * delta_t) - sin(particle.theta));
			change[1] = (velocity/yaw_rate) * (-cos(particle.theta + yaw_rate * delta_t) + cos(particle.theta));
		}
		change[2] = yaw_rate * delta_t;
		
		particle.x+= change[0] + dist_x(gen);
		particle.y+= change[1] + dist_y(gen);
		particle.theta+= change[2] + dist_theta(gen);
		particles[i] = particle;
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	// if(!observations.size()) return;
	for(int i=0; i<observations.size(); i++)
	{
		LandmarkObs obs = observations[i];
		double min_dist = 100000.0;
		int min_id = 0;

		for(int j=0; j<predicted.size(); j++)
		{
			LandmarkObs pred = predicted[j];
			double distance = sqrt(pow(pred.x - obs.x,2) + pow(pred.y - obs.y,2));
			if(distance<min_dist)
			{
				min_dist = distance;
				min_id = j;
			}
		}
		observations[i].id = min_id;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) 
{
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	double weight_sum = 0.0;

	for(int i=0; i<num_particles; i++)
	{
		Particle particle = particles[i];
		vector<LandmarkObs> predicted;

		//Sorting out the landmarks which are at a distance greater than the sensor range from the particle.
		for(int j=0; j<map_landmarks.landmark_list.size(); j++)
		{
			Map::single_landmark_s pred = map_landmarks.landmark_list[j];
			double distance = dist(pred.x_f, pred.y_f, particle.x, particle.y);
			if(distance<=sensor_range)
			{
				LandmarkObs landmark_in_range;
				landmark_in_range.set_values(pred.id_i, pred.x_f, pred.y_f);
				predicted.push_back(landmark_in_range);
			} 
		}

		//Transformation of the lidar observations to the global coordinates from the vehicle coordinate system.//Transformation of the lidar observations to the global coordinates from the vehicle coordinate system.
		vector<LandmarkObs> trans_observations = observations;
		for(int k=0; k<trans_observations.size(); k++)
		{
			trans_observations[k].x = particle.x + cos(particle.theta) * observations[k].x - sin(particle.theta) * observations[k].y;
			trans_observations[k].y = particle.y + sin(particle.theta) * observations[k].x + cos(particle.theta) * observations[k].y;
		}

		//Data Association
		dataAssociation(predicted, trans_observations);
		
		//Setting the associations
		for(int m=0; m<trans_observations.size(); m++)
		{
			particles[i].associations.push_back(predicted[trans_observations[m].id].id);
			particles[i].sense_x.push_back(trans_observations[m].x);
			particles[i].sense_y.push_back(trans_observations[m].y);
		}

		//Evaluating the probability of the vehicle being at a particle's posiiton.
		double prob = 1.0;

		for(int l=0; l<trans_observations.size(); l++)
		{
			double x = trans_observations[l].x;
			double y = trans_observations[l].y;
			double x_mean = predicted[trans_observations[l].id].x;
			double y_mean = predicted[trans_observations[l].id].y;

			prob*=gauss_prob(x, x_mean, std_landmark[0], y, y_mean, std_landmark[1]);
		}
		weights[i] = prob;
		particles[i].weight = prob;
		weight_sum+= prob;
	}

	for(int i=0; i<num_particles; i++)
	{
		weights[i] /= weight_sum;
		particles[i].weight = weights[i];
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	//Normalizing the weights.
	double weight_max = *max_element(begin(weights), end(weights));

	//Resampling the particles using the resampling wheel algorithm.
	uniform_int_distribution<int> index_dist(0,num_particles-1);
	uniform_real_distribution<double> beta_dist(0.0, 2 * weight_max);

	int index = index_dist(gen);//check for auto keyword.
	double beta = 0.0;
	vector<Particle> resampled_particles;
	
	for(int i=0; i<num_particles; i++)
	{
		beta+= beta_dist(gen);
		while(beta > weights[index])
		{
			beta-= weights[index];
			index+=1;
			index = index%(num_particles);
		}
		resampled_particles.push_back(particles[index]);
	}
	particles = resampled_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
    return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
