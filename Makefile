CC = gcc
CFLAGS = -Wall -Wextra -std=c99 -O2
LDFLAGS = -lm

TARGET = neural_net_demo
SOURCES = neural_net.c test.c
OBJECTS = $(SOURCES:.c=.o)
HEADERS = neural_net.h

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) $(OBJECTS) -o $(TARGET) $(LDFLAGS)

%.o: %.c $(HEADERS)
	$(CC) $(CFLAGS) -c $< -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(OBJECTS) $(TARGET)

# Debug build
debug: CFLAGS += -g -DDEBUG
debug: clean $(TARGET)