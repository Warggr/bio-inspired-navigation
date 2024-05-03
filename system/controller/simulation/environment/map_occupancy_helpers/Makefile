FLAGS = -O3 -Wall -fPIC -shared -std=c++11 \
	-static-libstdc++ \
    $(shell python3 -m pybind11 --includes) \
    -Wl,-rpath,"$$ORIGIN"

PYTHONVER = $(shell python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" )
extension = $(shell python${PYTHONVER}-config --extension-suffix)

all: map_utils_cpp$(extension) math_utils_cpp$(extension)

map_utils_cpp$(extension): map_utils.cpp
	g++ $(FLAGS) $< -o $@

math_utils_cpp$(extension): math_utils.cpp
	g++ $(FLAGS) $< -o $@

.PHONY: all
