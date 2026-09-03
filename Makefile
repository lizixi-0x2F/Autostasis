CC      = gcc
CFLAGS  = -O2 -Wall -Wextra -std=c11 -D_POSIX_C_SOURCE=199309L
BIN     = c/autostasis
OBJS    = c/term.o c/eval.o c/main.o

$(BIN): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $(OBJS)

c/%.o: c/%.c c/term.h
	$(CC) $(CFLAGS) -c -o $@ $<

run: $(BIN)
	./$(BIN)

clean:
	rm -f $(OBJS) $(BIN)
