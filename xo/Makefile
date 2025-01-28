NAME=ai
CXX=g++
CXXFLAGS=-std=c++17  -pedantic -O3
# LDFLAGS=-lSDL2 -lSDL2_image -lSDL2_ttf -lSDL2_mixer

SRC_FILES=$(wildcard *.cpp)
OBJ_FILES=$(patsubst %.cpp,%.o,$(SRC_FILES))

all: $(NAME)

$(NAME): $(OBJ_FILES)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)
	
%.o: %.cpp %.hpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<
	
clean:
	rm -f $(NAME) $(OBJ_FILES)

run: $(NAME)
	./$(NAME)
	
debug: CXXFLAGS += -g
debug: $(NAME)

re: clean all

.PHONY: all clean