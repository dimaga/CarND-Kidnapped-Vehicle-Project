/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *	  Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <random>
#include <cmath>
#include <cassert>
#include <list>
#include <vector>

#include "particle_filter.h"

namespace {
double normPi(double angleRad) {
	angleRad = std::fmod(angleRad, 2 * M_PI);
	if (angleRad > M_PI) {
		angleRad -= 2 * M_PI;
	}

	if (angleRad < -M_PI) {
		angleRad += 2 * M_PI;
	}

	return angleRad;
}

struct DistanceEntry {
	double distance_sq;
	int predicted_idx;
	int observed_idx;

	bool operator<(const DistanceEntry& other) const {
		return distance_sq < other.distance_sq;
	}
};

}


void ParticleFilter::init(double x, double y, double theta, double std[]) {
	Particle p;
	p.weight = 1.0;

	std::mt19937 gen;
	std::normal_distribution<double> N_x(0, std[0]);
	std::normal_distribution<double> N_y(0, std[1]);
	std::normal_distribution<double> N_theta(0, std[2]);

	num_particles = 100;
	for(int i = 0; i < num_particles; ++i) {
		p.id = i;
		p.x = x + N_x(gen);
		p.y = y + N_y(gen);
		p.theta = normPi(theta + N_theta(gen));

		particles.push_back(p);
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	using std::sin;
	using std::cos;

	std::mt19937 gen;
	std::normal_distribution<double> N_x(0, std_pos[0]);
	std::normal_distribution<double> N_y(0, std_pos[1]);
	std::normal_distribution<double> N_theta(0, std_pos[2]);

	for(int i = 0; i < num_particles; ++i) {
		Particle& p = particles.at(i);

		// Randomizing before and after prediction gives better filter performance
		p.theta = normPi(p.theta + N_theta(gen) * 0.5);
		p.x += N_x(gen) * 0.5;
		p.y += N_y(gen) * 0.5;
	}

	if (std::abs(yaw_rate) > 1e-5) {
		const double v_yr = velocity / yaw_rate;

		for(int i = 0; i < num_particles; ++i) {
			Particle& p = particles.at(i);

			p.x += v_yr * (sin(p.theta + yaw_rate * delta_t) - sin(p.theta));
			p.y += v_yr * (-cos(p.theta + yaw_rate * delta_t) + cos(p.theta));
		}
	} else {
		for(int i = 0; i < num_particles; ++i) {
			Particle& p = particles.at(i);

			p.x += velocity * cos(p.theta) * delta_t;
			p.y += velocity * sin(p.theta) * delta_t;
		}
	}

	for(int i = 0; i < num_particles; ++i) {
		Particle& p = particles.at(i);
		p.theta = normPi(p.theta + yaw_rate * delta_t);

		// Randomizing before and after prediction gives better filter performance
		p.theta = normPi(p.theta + N_theta(gen) * 0.5);
		p.x += N_x(gen) * 0.5;
		p.y += N_y(gen) * 0.5;
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	assert(!predicted.empty());

  	// Keep all distances in a table, start associations with best matches
	std::vector< DistanceEntry > distances_sq;

	for (std::size_t oi = 0, oi_count = observations.size(); oi < oi_count; ++oi) {
		LandmarkObs& obs = observations.at(oi);
		obs.id = -1;

		for (std::size_t pi = 0, pi_count = predicted.size(); pi < pi_count; ++pi) {
	  		LandmarkObs& pred = predicted.at(pi);
		  	pred.id = -1;

			DistanceEntry d;
			d.distance_sq = (pred.x - obs.x) * (pred.x - obs.x) + (pred.y - obs.y) * (pred.y - obs.y);
			d.predicted_idx = pi;
		    d.observed_idx = oi;
			distances_sq.push_back(d);
		}
  	}

	std::sort(distances_sq.begin(), distances_sq.end());

	for (std::size_t di = 0, di_count = distances_sq.size(); di < di_count; ++di) {
		const DistanceEntry& distance_entry = distances_sq.at(di);
		if (distance_entry.distance_sq > 7 * 7) {
			// All other links are probably outliers
			break;
		}

	  	LandmarkObs& obs = observations.at(distance_entry.observed_idx);
		LandmarkObs& pred = predicted.at(distance_entry.predicted_idx);
	  	if (-1 == obs.id && -1 == pred.id) {
			obs.id = distance_entry.predicted_idx;
			pred.id = distance_entry.observed_idx;
		}
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {

	std::vector<LandmarkObs> predicted;

	for (int pi = 0; pi < num_particles; ++pi) {
		Particle& p = particles.at(pi);

		predicted.clear();
		for (std::size_t li = 0, li_count = map_landmarks.landmark_list.size(); li < li_count; ++li) {
			const Map::single_landmark_s& single_landmark = map_landmarks.landmark_list.at(li);

			LandmarkObs obs;

			const double tx = single_landmark.x_f - p.x;
			const double ty = single_landmark.y_f - p.y;
			if (tx * tx + ty * ty > sensor_range * sensor_range) {
				// This landmark will not be visible, exclude it from predicted list for better runtime performance
				continue;
			}
		  
			const double cos_theta = std::cos(p.theta);
			const double sin_theta = std::sin(p.theta);

			obs.x = tx * cos_theta + ty * sin_theta;
			obs.y = -tx * sin_theta + ty * cos_theta;

			predicted.push_back(obs);
		}

		if (predicted.empty()) {
			// Exclude that particle as highly unlikely
			p.weight = 0.0;
			continue;
		}

		dataAssociation(predicted, observations);

		// I have removed normalization constant from Gaussian, since weights do not have to be normalized
		// Also, for better numerical robustness and performance, instead of product of exponents, I use
		// sum of logaritms
		double neg_log_w = 0;
		for (std::size_t oi = 0, oi_count = observations.size(); oi < oi_count; ++oi) {
			const LandmarkObs &obs = observations.at(oi);

		  	if (-1 == obs.id) {
			  	// No good association, use sensor range as maximum error
				neg_log_w += 0.5 * (0.5 * sensor_range * sensor_range / std_landmark[0]);
				neg_log_w += 0.5 * (0.5 * sensor_range * sensor_range / std_landmark[1]);
			} else {
				const LandmarkObs& pred = predicted.at(obs.id);
				neg_log_w += 0.5 * ((obs.x - pred.x) * (obs.x - pred.x) / std_landmark[0]);
				neg_log_w += 0.5 * ((obs.y - pred.y) * (obs.y - pred.y) / std_landmark[1]);
			}
		}

		p.weight = std::exp(-neg_log_w);
	}
}

void ParticleFilter::resample() {
	std::vector<double> weights;
	for (int pi = 0; pi < num_particles; ++pi) {
		const Particle &p = particles.at(pi);
		weights.push_back(p.weight);
	}

	std::vector<Particle> new_particles;

	std::mt19937 gen;
	std::discrete_distribution<> d(weights.begin(), weights.end());
	for (int pi = 0; pi < num_particles; ++pi) {
		Particle new_particle = particles.at(d(gen));
		new_particle.weight = 1.0;

		new_particles.push_back(new_particle);
	}

    particles = new_particles;
}

void ParticleFilter::write(std::string filename) {
	// You don't need to modify this file.
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
}
