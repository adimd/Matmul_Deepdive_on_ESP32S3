idf_component_register(
    SRCS
        "main.c"
        "matmul_naive.c"
    INCLUDE_DIRS "."
)

# Per-file optimization overrides
# matmul_naive.c → -O0 so we get truly unoptimized baseline
set_source_files_properties(matmul_naive.c PROPERTIES
    COMPILE_FLAGS "-O0 -fno-inline"
)
