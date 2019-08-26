rebuild: clean all

all: behavioral.so

clean:
	rm -f *.o *.so *.html behavioral.hpp.gch

behavioral.so: behavioral.o behavioral.hpp
	g++ -shared -g -w -o behavioral.so behavioral.o carry_chain.o lut.o

behavioral.o: behavioral.cpp behavioral.hpp
	g++ -Wall -fpic -g -w -c *.cpp

