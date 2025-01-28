#ifndef __GAME_STATE_HPP__
#define __GAME_STATE_HPP__

#include <iostream> // IWYU pragma: keep
#include <vector>
#include <string> // IWYU pragma: keep

class GameState {
private:
    int board[3][3];
    int turn;        // 1 for X, -1 for O
    int winner;      // 0 for no winner, 1 for X, -1 for O
    int moves;
    bool gameOver;

public:
    GameState() {
        reset();
    }

    void reset() {
        for(int i = 0; i < 3; i++)
            for(int j = 0; j < 3; j++)
                board[i][j] = 0;
        turn = 1;
        winner = 0;
        moves = 0;
        gameOver = false;
    }

    bool makeMove(int x, int y) {
        if(x < 0 || x > 2 || y < 0 || y > 2 || board[x][y] != 0 || gameOver)
            return false;
        
        board[x][y] = turn;
        moves++;
        checkWinner();
        turn = -turn;
        return true;
    }

    std::vector<double> getBoardState() {
        std::vector<double> state;
        for(int i = 0; i < 3; i++)
            for(int j = 0; j < 3; j++)
                state.push_back(board[i][j]);
        return state;
    }

    bool isGameOver() { return gameOver; }
    int getWinner() { return winner; }
    int getCurrentPlayer() { return turn; }

private:
    void checkWinner() {
        // Check rows
        for(int i = 0; i < 3; i++) {
            if(board[i][0] != 0 && board[i][0] == board[i][1] && board[i][1] == board[i][2]) {
                winner = board[i][0];
                gameOver = true;
                return;
            }
        }
        // Check columns
        for(int j = 0; j < 3; j++) {
            if(board[0][j] != 0 && board[0][j] == board[1][j] && board[1][j] == board[2][j]) {
                winner = board[0][j];
                gameOver = true;
                return;
            }
        }
        // Check diagonals
        if(board[0][0] != 0 && board[0][0] == board[1][1] && board[1][1] == board[2][2]) {
            winner = board[0][0];
            gameOver = true;
            return;
        }
        if(board[0][2] != 0 && board[0][2] == board[1][1] && board[1][1] == board[2][0]) {
            winner = board[0][2];
            gameOver = true;
            return;
        }
        // Check draw
        if(moves == 9) {
            gameOver = true;
        }
    }
};

#endif