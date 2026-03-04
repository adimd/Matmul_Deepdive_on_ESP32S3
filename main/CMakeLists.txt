idf_component_register(
    SRCS
        "main.c"
        "matmul_naive.c"
        "matmul_opt.c"
        "matmul_tiled.c"
    INCLUDE_DIRS "."
)

set_source_files_properties(matmul_naive.c PROPERTIES
    COMPILE_FLAGS "-O0 -fno-inline"
)

set_source_files_properties(matmul_opt.c PROPERTIES
    COMPILE_FLAGS "-O3"
)

set_source_files_properties(matmul_tiled.c PROPERTIES
    COMPILE_FLAGS "-O3"
)