#ifndef __NEURAL_NETWORK_HPP__
#define __NEURAL_NETWORK_HPP__

#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <deque>
#include <random>
#include <fstream>
#include "json.hh"

#include "GameState.hpp"

using namespace std;
using json = nlohmann::json;

struct Experience {
    std::vector<double> state;
    int action;
    double reward;
    std::vector<double> nextState;
    bool gameOver;
};

class NeuralNetwork {
private:
    vector<vector<vector<double>>> weights; // layers x nodes x connections
    vector<vector<double>> biases;              // layers x nodes
    vector<vector<double>> neurons;             // layers x nodes
	// Q-learning parameters
    double learningRate = 0.001;
    double gamma = 0.95;
    std::deque<Experience> replayBuffer;
    const size_t maxReplaySize = 10000;
    const size_t miniBatchSize = 32;
	
	// Epsilon-greedy parameters
	double epsilon = 1.0;
    double epsilonMin = 0.01;
    double epsilonDecay = 0.995;
	
	// Training statistics
	int totalGames = 0;
    int winsCount = 0;
    vector<double> recentWinRates;
    
public:
    NeuralNetwork() {
        // Initialize with 9-18-9 architecture
        weights.resize(2);  // 2 layers of weights (input->hidden, hidden->output)
        biases.resize(2);   // 2 layers of biases
        neurons.resize(3);  // 3 layers of neurons (input, hidden, output)

        // Input layer (9 neurons)
        neurons[0].resize(9);
        
        // Hidden layer (18 neurons)
        neurons[1].resize(18);
        weights[0].resize(18, vector<double>(9));
        biases[0].resize(18);
        
        // Output layer (9 neurons)
        neurons[2].resize(9);
        weights[1].resize(9, vector<double>(18));
        biases[1].resize(9);

        randomizeWeights();
    }
	
	std::vector<int> getValidMoves(const std::vector<double>& state) {
		std::vector<int> valid;
		for(int i = 0; i < 9; i++) {
			if(state[i] == 0) valid.push_back(i);
		}
		return valid;
	}

	int getAction(const std::vector<double>& state) {
		auto validMoves = getValidMoves(state);
		if(validMoves.empty()) return -1;
		
		// Exploration
		if(((double)rand() / RAND_MAX) < epsilon) {
			return validMoves[rand() % validMoves.size()];
		}
		
		// Exploitation
		auto qValues = forward(state);
		int bestMove = -1;
		double bestValue = -std::numeric_limits<double>::infinity();
		
		for(int move : validMoves) {
			if(qValues[move] > bestValue) {
				bestValue = qValues[move];
				bestMove = move;
			}
		}
		return bestMove;
	}

    void decayEpsilon() {
        epsilon = max(epsilonMin, epsilon * epsilonDecay);
    }

    void randomizeWeights() {
        srand(time(NULL));
        for(auto& layer : weights)
            for(auto& node : layer)
                for(auto& weight : node)
                    weight = ((double)rand() / RAND_MAX) * 2 - 1;

        for(auto& layer : biases)
            for(auto& bias : layer)
                bias = ((double)rand() / RAND_MAX) * 2 - 1;
    }
	
	void updateQ(const Experience& exp) {
        // Current Q-values
        auto currentQ = forward(exp.state);
        
        // Target Q-values
        auto nextQ = forward(exp.nextState);
        double maxNextQ = *std::max_element(nextQ.begin(), nextQ.end());
        
        // Q-learning update
        double target = exp.reward;
        if (!exp.gameOver) {
            target += gamma * maxNextQ;
        }
        
        // Update only the Q-value for the taken action
        std::vector<double> targetQ = currentQ;
        targetQ[exp.action] = currentQ[exp.action] + 
                            learningRate * (target - currentQ[exp.action]);
        
        // Backpropagate
        backpropagate(exp.state, targetQ);
    }

    void addExperience(const Experience& exp) {
        if (replayBuffer.size() >= maxReplaySize) {
            replayBuffer.pop_front();
        }
        replayBuffer.push_back(exp);
    }
	
	void train(GameState& game, int numEpisodes) {
        for(int episode = 0; episode < numEpisodes; episode++) {
            game.reset();
            vector<double> currentState = game.getBoardState();
            bool done = false;
            
            // Play one complete game
            while(!done) {
                // Get action using epsilon-greedy
                int action = getAction(currentState);
                
                // Make move
                int x = action / 3;
                int y = action % 3;
                bool validMove = game.makeMove(x, y);
                
                if(!validMove) continue;
                
                // Get new state and reward
                vector<double> nextState = game.getBoardState();
                double reward = calculateReward(game);
                done = game.isGameOver();
                
                // Store experience
                Experience exp{currentState, action, reward, nextState, done};
                addExperience(exp);
                
                // Train on mini-batch
                train();
                
                currentState = nextState;
            }
            
            // Post-episode updates
            updateStats(game);
            decayEpsilon();
            totalGames++;
            
            // Print progress every 100 episodes
            if((episode + 1) % 100 == 0) {
                printStats(episode + 1);
            }
        }
    }
	
	 double calculateReward(GameState& game) {
        if(!game.isGameOver()) return 0.0;
        
        if(game.getWinner() == 1) return 1.0;  // Win
        if(game.getWinner() == -1) return -1.0; // Loss
        return 0.1; // Draw
    }
    
    void updateStats(GameState& game) {
        if(game.getWinner() == 1) winsCount++;
        
        // Track win rate over last 100 games
        if(totalGames % 100 == 0) {
            double winRate = winsCount / 100.0;
            recentWinRates.push_back(winRate);
            winsCount = 0;
        }
    }
    
    void printStats(int episode) {
        cout << "Episode: " << episode << endl;
        cout << "Epsilon: " << epsilon << endl;
        if(!recentWinRates.empty()) {
            cout << "Win Rate: " << recentWinRates.back() << endl;
        }
        cout << "-------------------" << endl;
    }

    void train() {
        if (replayBuffer.size() < miniBatchSize) return;
        
        // Sample random batch
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, replayBuffer.size() - 1);
        
        for (size_t i = 0; i < miniBatchSize; ++i) {
            updateQ(replayBuffer[dis(gen)]);
        }
    }

    vector<double> forward(const vector<double>& input) {
        // Set input layer
        neurons[0] = input;

        // Hidden layer
        for(int i = 0; i < 18; i++) {
            double sum = 0;
            for(int j = 0; j < 9; j++)
                sum += neurons[0][j] * weights[0][i][j];
            neurons[1][i] = tanh(sum + biases[0][i]);
        }

        // Output layer
        for(int i = 0; i < 9; i++) {
            double sum = 0;
            for(int j = 0; j < 18; j++)
                sum += neurons[1][j] * weights[1][i][j];
            neurons[2][i] = tanh(sum + biases[1][i]);
        }

        return neurons[2];
    }
	
	void backpropagate(const std::vector<double>& input, const std::vector<double>& target) {
        // Forward pass
        forward(input);
        
        // Output layer deltas
        std::vector<double> outputDeltas(9);
        for(int i = 0; i < 9; i++) {
            double output = neurons[2][i];
            outputDeltas[i] = (target[i] - output) * (1 - output * output);
        }
        
        // Hidden layer deltas
        std::vector<double> hiddenDeltas(18);
        for(int i = 0; i < 18; i++) {
            double sum = 0;
            for(int j = 0; j < 9; j++) {
                sum += outputDeltas[j] * weights[1][j][i];
            }
            hiddenDeltas[i] = sum * (1 - neurons[1][i] * neurons[1][i]);
        }
        
        // Update weights and biases
        // Output layer
        for(int i = 0; i < 9; i++) {
            for(int j = 0; j < 18; j++) {
                weights[1][i][j] += learningRate * outputDeltas[i] * neurons[1][j];
            }
            biases[1][i] += learningRate * outputDeltas[i];
        }
        
        // Hidden layer
        for(int i = 0; i < 18; i++) {
            for(int j = 0; j < 9; j++) {
                weights[0][i][j] += learningRate * hiddenDeltas[i] * neurons[0][j];
            }
            biases[0][i] += learningRate * hiddenDeltas[i];
        }
    }
	int selectRandomAction(const vector<double>& state) {
        vector<int> validMoves;
        for(int i = 0; i < 9; i++) {
            if(state[i] == 0) { // Empty cell
                validMoves.push_back(i);
            }
        }
        
        if(validMoves.empty()) return -1; // No valid moves
        
        int randomIndex = rand() % validMoves.size();
        return validMoves[randomIndex];
    }
	
	//load and save weights
	    void saveModel(const string& filename) {
        json j;
        
        // Save weights (3D array)
        j["weights"] = json::array();
        for(const auto& layer : weights) {
            json layerArray = json::array();
            for(const auto& node : layer) {
                layerArray.push_back(node);
            }
            j["weights"].push_back(layerArray);
        }
        
        // Save biases (2D array)
        j["biases"] = json::array();
        for(const auto& layer : biases) {
            j["biases"].push_back(layer);
        }
        
        // Save parameters
        j["epsilon"] = epsilon;
        j["totalGames"] = totalGames;
        j["recentWinRates"] = recentWinRates;
        
        ofstream file(filename);
        file << j.dump(4);
    }
    
    void loadModel(const string& filename) {
        ifstream file(filename);
        if (!file.is_open()) {
            throw runtime_error("Could not open file: " + filename);
        }
        
        json j;
        file >> j;
        
        // Load weights
        if (j.contains("weights")) {
            for(size_t i = 0; i < j["weights"].size() && i < weights.size(); i++) {
                for(size_t k = 0; k < j["weights"][i].size() && k < weights[i].size(); k++) {
                    weights[i][k] = j["weights"][i][k].get<vector<double>>();
                }
            }
        }
        
        // Load biases
        if (j.contains("biases")) {
            for(size_t i = 0; i < j["biases"].size() && i < biases.size(); i++) {
                biases[i] = j["biases"][i].get<vector<double>>();
            }
        }
        
        // Load parameters
        if (j.contains("epsilon")) epsilon = j["epsilon"].get<double>();
        if (j.contains("totalGames")) totalGames = j["totalGames"].get<int>();
        if (j.contains("recentWinRates")) recentWinRates = j["recentWinRates"].get<vector<double>>();
    }
};

#endif