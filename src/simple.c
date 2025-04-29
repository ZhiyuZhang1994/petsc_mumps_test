#include <petscmat.h>

/**
 * A =

   1   2   0   4
   0   1   2   0
   2   0   4   0
   0   1   2   1
 *
 *
 * P =

   1.00000   0.00000
   0.30000   0.50000
   0.00000   0.80000
   0.90000   0.00000
 *
 *
 */

int main(int argc, char *argv[])
{
    Mat         A, P;
    MPI_Comm    comm;
    PetscMPIInt rank, size;
    PetscFunctionBeginUser;
    PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));

    // // 构造数据 for two processor
    // PetscInt    i1[] = {0, 3, 5}, i2[] = {0, 2, 5};
    // PetscInt    j1[] = {0, 1, 3, 1, 2}, j2[] = {0, 2, 1, 2, 3};
    // PetscScalar a1[] = {1, 2, 4, 1, 2}, a2[] = {2, 4, 1, 2, 1};
    // PetscInt    pi1[] = {0, 1, 3}, pi2[] = {0, 1, 2};
    // PetscInt    pj1[] = {0, 0, 1}, pj2[] = {1, 0};
    // PetscScalar pa1[] = {1, 0.3, 0.5}, pa2[] = {0.8, 0.9};
    // comm = PETSC_COMM_WORLD;
    // PetscCallMPI(MPI_Comm_rank(comm, &rank));
    // PetscCallMPI(MPI_Comm_size(comm, &size));
    // PetscCheck(size == 2, comm, PETSC_ERR_WRONG_MPI_SIZE, "You have to use two processor cores to run this example ");
    // PetscCall(MatCreateMPIAIJWithArrays(comm, 2, 2, PETSC_DETERMINE, PETSC_DETERMINE, rank ? i2 : i1, rank ? j2 : j1, rank ? a2 : a1, &A));
    // PetscCall(MatCreateMPIAIJWithArrays(comm, 2, 1, PETSC_DETERMINE, PETSC_DETERMINE, rank ? pi2 : pi1, rank ? pj2 : pj1, rank ? pa2 : pa1, &P));


    // construct data for one processor
    PetscInt    i1[] = {0, 3, 5, 7, 10};
    PetscInt    j1[] = {0, 1, 3, 1, 2, 0, 2, 1, 2, 3};
    PetscScalar a1[] = {1, 2, 4, 1, 2, 2, 4, 1, 2, 1};
    PetscInt    pi1[] = {0, 1, 3, 4, 5};
    PetscInt    pj1[] = {0, 0, 1, 1, 0};
    PetscScalar pa1[] = {1, 0.3, 0.5, 0.8, 0.9};
    comm = PETSC_COMM_WORLD;
    PetscCallMPI(MPI_Comm_size(comm, &size));
    PetscCheck(size == 1, comm, PETSC_ERR_WRONG_MPI_SIZE, "You have to use one processor cores to run this example ");
    PetscCall(MatCreateSeqAIJWithArrays(comm, 4, 4, i1, j1, a1, &A));
    PetscCall(MatCreateSeqAIJWithArrays(comm, 4, 2, pi1, pj1, pa1, &P));

    MatView(A, PETSC_VIEWER_STDOUT_WORLD);
    MatView(P, PETSC_VIEWER_STDOUT_WORLD);
    // 求解
    Mat A_lu;
    Mat X; // ans
    IS row, col;
    MatFactorInfo info;
    MatFactorInfoInitialize(&info);
    MatGetFactor(A, MATSOLVERMUMPS, MAT_FACTOR_LU, &A_lu);
    MatGetOrdering(A, MATORDERINGEXTERNAL, &row, &col);
    MatLUFactorSymbolic(A_lu, A, row, col, &info);
    MatLUFactorNumeric(A_lu, A, &info);
    MatMatSolve(A_lu, P, X);


    // deconstruct
    PetscCall(MatDestroy(&A));
    PetscCall(MatDestroy(&P));
    PetscCall(MatDestroy(&X));
    PetscCall(PetscFinalize());
  return 0;
}
