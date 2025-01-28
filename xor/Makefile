NAME = nn
HEADERS = Neuron.hpp Layer.hpp Matrix.hpp NeuralNetwork.hpp
SRC_FILES = main.cpp Neuron.cpp Layer.cpp Matrix.cpp NeuralNetwork.cpp
OBJ_FILES = $(SRC_FILES:.cpp=.o)

CXX = c++
CXXFLAGS = -std=c++11 -pedantic -O3
LDFLAGS = -lm

all: $(NAME) clean

$(NAME): $(OBJ_FILES)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)
	
%.o: %.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c -o $@ $<
	
clean:
	rm -f $(OBJ_FILES)
	
fclean: clean
	rm -f $(NAME)
	
re: fclean all
	
.PHONY: all clean