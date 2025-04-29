#include <petscmat.h>

/**
 * A =

   1   2   0   4
   2   1   0   0
   0   0   4   0
   4   0   0   1
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
    // PetscInt    i1[] = {0, 3, 5}, i2[] = {0, 1, 3};
    // PetscInt    j1[] = {0, 1, 3, 0, 1}, j2[] = {2, 0, 3};
    // PetscScalar a1[] = {1, 2, 4, 2, 1}, a2[] = {4, 4, 1};
    // PetscInt    pi1[] = {0, 1, 3}, pi2[] = {0, 1, 2};
    // PetscInt    pj1[] = {0, 0, 1}, pj2[] = {1, 0};
    // PetscScalar pa1[] = {1, 0.3, 0.5}, pa2[] = {0.8, 0.9};
    // comm = PETSC_COMM_WORLD;
    // PetscCallMPI(MPI_Comm_rank(comm, &rank));
    // PetscCallMPI(MPI_Comm_size(comm, &size));
    // PetscCheck(size == 2, comm, PETSC_ERR_WRONG_MPI_SIZE, "You have to use two processor cores to run this example ");
    // PetscCall(MatCreateMPIAIJWithArrays(comm, 2, 2, PETSC_DETERMINE, PETSC_DETERMINE, rank ? i2 : i1, rank ? j2 : j1, rank ? a2 : a1, &A));
    // PetscCall(MatCreateMPIAIJWithArrays(comm, 2, 1, PETSC_DETERMINE, PETSC_DETERMINE, rank ? pi2 : pi1, rank ? pj2 : pj1, rank ? pa2 : pa1, &P));
    // Mat P_dense; // note: MatMatSolve使用mumps时右端项必须是denseMat
    // MatConvert(P, MATDENSE, MAT_INITIAL_MATRIX, &P_dense);

    // construct data for one processor
    PetscInt    i1[] = {0, 3, 5, 6, 8};
    PetscInt    j1[] = {0, 1, 3, 0, 1, 2, 0, 3};
    PetscScalar a1[] = {1, 2, 4, 2, 1, 4, 4, 1};
    PetscInt    pi1[] = {0, 1, 3, 4, 5};
    PetscInt    pj1[] = {0, 0, 1, 1, 0};
    PetscScalar pa1[] = {1, 0.3, 0.5, 0.8, 0.9};
    comm = PETSC_COMM_WORLD;
    PetscCallMPI(MPI_Comm_size(comm, &size));
    PetscCheck(size == 1, comm, PETSC_ERR_WRONG_MPI_SIZE, "You have to use one processor cores to run this example ");
    PetscCall(MatCreateSeqAIJWithArrays(comm, 4, 4, i1, j1, a1, &A));
    PetscCall(MatCreateSeqAIJWithArrays(comm, 4, 2, pi1, pj1, pa1, &P));
    Mat P_dense; // note: MatMatSolve使用mumps时右端项必须是denseMat
    MatConvert(P, MATDENSE, MAT_INITIAL_MATRIX, &P_dense);

    // view input
    MatView(A, PETSC_VIEWER_STDOUT_WORLD);
    MatView(P_dense, PETSC_VIEWER_STDOUT_WORLD);

    // 求解
    Mat A_chol;
    Mat X; // ans
    MatDuplicate(P_dense, MAT_DO_NOT_COPY_VALUES, &X); // note: MatMatSolve使用mumps时解必须显式指定为denseMat
    IS row, col;
    MatFactorInfo info;
    MatFactorInfoInitialize(&info);
    MatGetFactor(A, MATSOLVERMUMPS, MAT_FACTOR_CHOLESKY, &A_chol);
    MatMumpsSetIcntl(A_chol, 29, 2); // 指定排序方法
    MatGetOrdering(A, MATORDERINGEXTERNAL, &row, &col);
    MatCholeskyFactorSymbolic(A_chol, A, row, &info);
    MatCholeskyFactorNumeric(A_chol, A, &info);
    MatMatSolve(A_chol, P_dense, X);

    // view ans
    MatView(X, PETSC_VIEWER_STDOUT_WORLD);
    printf("done\n");

    // deconstruct
    PetscCall(MatDestroy(&A));
    PetscCall(MatDestroy(&P));
    PetscCall(MatDestroy(&P_dense));
    PetscCall(MatDestroy(&X));
    PetscCall(PetscFinalize());
  return 0;
}
