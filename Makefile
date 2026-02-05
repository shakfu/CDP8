
.PHONY: all build clean libcdp cycdp test

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

# Build cycdp (Python bindings) - requires libcdp
cycdp: libcdp
	@cd cycdp && make build

# Run all tests
test: libcdp
	@cd libcdp/build && ./test_cdp
	@cd cycdp && make test

clean:
	@rm -rf build
	@rm -rf libcdp/build
	@cd cycdp && make clean 2>/dev/null || true


