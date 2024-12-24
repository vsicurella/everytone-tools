CC = gcc
CFLAGS = 

FAREY_DIR = ./c_farey
FAREY_TARGET = $(FAREY_DIR)/farey.so

all: $(FAREY_TARGET)

$(FAREY_TARGET): $(FAREY_DIR)/farey.c
	$(CC) -shared $(CFLAGS) -o $@ $<

clean:
	rm $(FAREY_TARGET)
