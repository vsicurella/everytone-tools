CC = gcc
CFLAGS = -Wall -g0

UTILS_DIR = ./pyc_utils
UTILS_TARGET = $(UTILS_DIR)/utils.o

all: $(UTILS_TARGET)

$(UTILS_TARGET): $(UTILS_DIR)/utils.c
	$(CC) $(CFLAGS) -arch arm64 -c $< -o $@ 

clean:
	rm $(UTILS_TARGET)
