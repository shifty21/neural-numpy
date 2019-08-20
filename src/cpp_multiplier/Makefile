all: behavioral.so

clean:
	rm -f *.o *.so *.html

behavioral.so: behavioral.o
	g++ -shared -g -o behavioral behavioral.o carry_chain.o lut.o

behavioral.o: behavioral.cpp
	g++ -Wall -fpic -g -c behavioral.cpp behavioral.hpp carry_chain.cpp lut.cpp

