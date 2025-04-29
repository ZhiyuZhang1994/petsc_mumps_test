aux_source_directory(. source_list)
add_executable(test     ${source_list})

target_link_libraries(test PUBLIC
    petsc
    dmumps
    mumps_common
    pord
    smumps
    parmetis
    metis
    y
    hwloc
    scalapack
    quadmath
    MPI::MPI_C          # MPI C 库， # todo: 采用${MPI_C_LIBRARIES}会失败，为什么
    openblas
    gfortran
    pthread
    OpenMP::OpenMP_C    # OpenMP C 库, todo: 是否要包含OpenMP::OpenMP_CXX
    stdc++              # C++ 标准库
    regex               # 正则表达式库
)

