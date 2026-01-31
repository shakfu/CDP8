
.PHONY: all build clean libcdp pycdp test

all: build

build:
	@mkdir -p build && cd build && \
		cmake .. -DCMAKE_POLICY_VERSION_MINIMUM=3.5 && \
		cmake --build . --config Release

# Build libcdp (C library)
libcdp:
	@mkdir -p libcdp/build && cd libcdp/build && \
		cmake .. && \
		cmake --build .

# Build pycdp (Python bindings) - requires libcdp
pycdp: libcdp
	@cd pycdp && make build

# Run all tests
test: libcdp
	@cd libcdp/build && ./test_cdp
	@cd pycdp && make test

clean:
	@rm -rf build
	@rm -rf libcdp/build
	@cd pycdp && make clean 2>/dev/null || true


