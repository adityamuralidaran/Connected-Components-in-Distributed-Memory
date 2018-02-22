CXX=mpicxx
CXXFLAGS=-std=c++14 -O3 -Wall

all: a1

demo:
	$(CXX) -DDEBUG $(CXXFLAGS) a1.cpp -o a1
	mpiexec -np 4 ./a1 8 6 out

clean:
	rm -rf a1
