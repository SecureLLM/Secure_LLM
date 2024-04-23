SRCS=$(wildcard *.cpp)
OBJS=$(SRCS:.cpp=.o)

all: $(OBJS)

%.o: %.cpp
	gcc -g -O -c $< -o $@ || (echo "Deleting $<" && echo "$<" >> rejected_files.log && mv $< $<.reject)
