#ifndef MATRIXUTIL_HPP
#define MATRIXUTIL_HPP

#ifdef __cplusplus
extern "C"{
#endif
    __global__ void matrixMult(double* a, double* b, double* c, int width, int C_rows, int C_cols);
    __global__ void matrixSum(double* a, double* b, double* c, int x, int y);
    __global__ void matrixDif(double* a, double* b, double* c, int x, int y);
    __global__ void matrixSigmoid(double* a, int x, int y);
    __global__ void matrixTrasponi(double* a, double* b, int x, int y);
    __global__ void matrixMultInt(double* a, double b, double* c, int x, int y);
    __global__ void matrixSumInt(double* a, double b, double* c, int x, int y);
    __global__ void matrixDifInt(double* a, double b, double* c, int x, int y);
#ifdef __cplusplus
}
#endif

#endif