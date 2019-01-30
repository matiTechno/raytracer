all:
	g++ -std=c++14 -Wall -Wextra -o raytracer main.cpp -DNDEBUG -O3 -fno-rtti -fno-exceptions -pthread -g -fno-omit-frame-pointer
