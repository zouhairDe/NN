#include <iostream>
#include "NeuralNetwok.hpp"
#include "GameState.hpp"



void displayBoard(GameState& game) {
    std::vector<double> state = game.getBoardState();
    std::cout << "\n";
    for(int i = 0; i < 3; i++) {
        std::cout << " ";
        for(int j = 0; j < 3; j++) {
            char symbol = ' ';
            if(state[i*3 + j] == 1) symbol = 'X';
            if(state[i*3 + j] == -1) symbol = 'O';
            std::cout << symbol;
            if(j < 2) std::cout << " | ";
        }
        std::cout << "\n";
        if(i < 2) std::cout << "-----------\n";
    }
    std::cout << "\n";
}

void humanVsAI(NeuralNetwork& network) {
    GameState game;
    bool humanTurn = true;
    
    while(!game.isGameOver()) {
        displayBoard(game);
        auto state = game.getBoardState();
        
        if(humanTurn) {
            int x, y;
            std::cout << "Enter move (row[0-2] col[0-2]): ";
            std::cin >> x >> y;
            if(!game.makeMove(x, y)) {
                std::cout << "Invalid move!\n";
                continue;
            }
        } else {
            int action = network.getAction(state);
            if(action == -1 || !game.makeMove(action/3, action%3)) {
                std::cout << "AI made invalid move!\n";
                continue;
            }
            std::cout << "AI plays: " << action/3 << " " << action%3 << "\n";
        }
        humanTurn = !humanTurn;
    }
}



void selfPlay(NeuralNetwork& network, int games = 100) {
    GameState game;
    int aiWins = 0, draws = 0;
    
    for(int i = 0; i < games; i++) {
        game.reset();
        while(!game.isGameOver()) {
            auto state = game.getBoardState();
            int action = network.getAction(state);
            game.makeMove(action/3, action%3);
        }
        
        if(game.getWinner() == 1) aiWins++;
        else if(game.getWinner() == 0) draws++;
        
        if((i+1) % 10 == 0) {
            std::cout << "Games played: " << i+1 << "\n";
            std::cout << "AI wins: " << aiWins << "\n";
            std::cout << "Draws: " << draws << "\n";
            std::cout << "----------------\n";
        }
    }
}

void	displayBoardIndex(){
	std::cout << "\n";
	for(int i = 0; i < 3; i++) {
		std::cout << " ";
		for(int j = 0; j < 3; j++) {
			std::cout << i*3 + j;
			if(j < 2) std::cout << " | ";
		}
		std::cout << "\n";
		if(i < 2) std::cout << "-----------\n";
	}
	std::cout << "\n";
}

int main() {
    NeuralNetwork network;
    GameState trainingGame;
    
    // Load previous model if exists
    try {
        network.loadModel("model_final.json");
        std::cout << "Loaded previous model. Continuing training...\n";
		system("rm model_checkpoint_*.json model_final.json");
    } catch(...) {
        std::cout << "Starting fresh training...\n";
    }
    
    // Train in stages
    for(int i = 0; i < 5; i++) {
        network.train(trainingGame, 5000000);
        network.saveModel("model_checkpoint_" + std::to_string(i) + ".json");
        std::cout << "Completed training stage " << i + 1 << "/5\n";
    }
    
    network.saveModel("model_final.json");
    // Test scenarios
    char choice;
	displayBoardIndex();
    do {
        std::cout << "\n1. Play against AI\n";
        std::cout << "2. Watch AI self-play\n";
        std::cout << "3. Exit\n";
        std::cout << "Choice: ";
        std::cin >> choice;
        
        switch(choice) {
            case '1':
                humanVsAI(network);
                break;
            case '2':
                selfPlay(network);
                break;
            case '3':
                return 0;
        }
    } while(true);
    
    return 0;
}