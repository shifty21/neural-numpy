all: convo_multiplier.so

clean:
	rm -f *.o *.so *.html

convo_multiplier.so: convo_multiplier.o
	g++ -shared -g -o libclib1.so convo_multiplier.o

convo_multiplier.o: convo_multiplier.cpp
	g++ -Wall -fpic -g -c convo_multiplier.cpp
