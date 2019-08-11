all: clib1.so

clean:
	rm -f *.o *.so *.html

clib1.so: clib1.o
	g++ -shared -o libclib1.so clib1.o

clib1.o: clib1.cpp
	g++ -Wall -fpic -c clib1.cpp
