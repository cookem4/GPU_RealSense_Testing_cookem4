NAME=houghLines
INC_DIRS=include
INC_DIRS+=/usr/local/lib
INC_DIRS+=/usr/bin/local/include
OBJ_DIR=obj

CC=g++
SOURCES=houghlines.cpp
SOURCES+=rsCam.cpp
CFLAGS+=-Wall -g -std=gnu++11

LDLIBS=-L/usr/local/lib/

OPENCV=$(shell pkg-config opencv --cflags --libs)
LDLIBS+=-lrealsense2 $(OPENCV)
LDLIBS+=-lpthread
WITH_CUDA=ON
OBJ=$(filter %.o,$(SOURCES:%.cpp=$(OBJ_DIR)/%.o))

vpath %.hpp $(INC_DIRS)
vpath %.cpp $(SRC_DIRS)

.PHONY: all
all: $(NAME)

$(NAME): $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

$(OBJ): $(OBJ_DIR)/%.o: %.cpp | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c -o $@ $<

$(OBJ_DIR):
	@mkdir -p $(OBJ_DIR)

.PHONY: clean
clean:
	rm -rf $(OBJ_DIR) $(NAME)

