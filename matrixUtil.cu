#include <cmath>
#include "matrixUtil.hpp"

// funzione per moltiplicare due matrici
// x è il numero di righe di a
// y è il numero di colonne di b
// z è il numero di colonne di a e righe di b, se non sono uguali non si può effettuare la moltiplicazione
__global__ void matrixMult(double* a, double* b, double* c, int width, int C_rows, int C_cols){
    int row = blockIdx.y * blockDim.y + threadIdx.y;   
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    // check boundry conditions
    if( row < C_rows && col < C_cols ){
        // do the multiplication for one row and col
        double value = 0;
        for(int k = 0; k < width; k++){
        value += a[row * width + k] * b[k * C_cols + col];
        }
        // store result
        c[row * C_cols + col] = value;
    }
    
}
// funzione per sommare due matrici
// le due matrici devono avere la stessa dimensione
// il risultato viene messo nella terza matrice
__global__ void matrixSum(double* a, double* b, double* c, int x, int y){
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    if(row < y && col < x){
        c[col * y + row] = a[col * y + row] + b[col * y + row];
    }
}
// funzione per sottrarre due matrici
// le due matrici devono avere la stessa dimensione
// il risultato viene messo nella terza matrice
__global__ void matrixDif(double* a, double* b, double* c, int x, int y){
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    if(row < y && col < x){
        c[col * y + row] = a[col * y + row] - b[col * y + row];
    }
}
// funzione per applicare la sigmoide
// il risultato viene messo nella stessa matrice
__global__ void matrixSigmoid(double* a, int x, int y){
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    if(row < y && col < x){
        a[col * y + row] = 1.0 / (1.0 + __expf(-a[col * y + row]));
    }
}
// funzione per trasporre la matrice
// il risultato viene messo nella seconda matrice
// fai attenzione che abbiano le stesse dimensioni
__global__ void matrixTrasponi(double* a, double* b, int x, int y){
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    if(row < y && col < x){
        b[col * y + row] = a[row * x + col];
    }
}
// funzione per moltiplicare una matrice con un numero intero
// la x e la y sono della matrice iniziale
// la terza matrice deve avere dimensioni x * y e il risultato verrà messo in quella
__global__ void matrixMultInt(double* a, double b, double* c, int x, int y){
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    if(row < y && col < x){
        c[col * y + row] = a[col * y + row] * b;
    }
}
// funzione per sommare una matrice con un numero intero
// la x e la y sono della matrice
// la seconda matrice avrà i risultati
__global__ void matrixSumInt(double* a, double b, double* c, int x, int y){
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    if(row < y && col < x){
        c[col * y + row] = a[col * y + row] + b;
    }
}
// funzione per sottrarre da una matrice un numero intero
// la x e la y sono della matrice
// la seconda matrice avrà i risultati
__global__ void matrixDifInt(double* a, double b, double* c, int x, int y){
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    if(row < y && col < x){
        c[col * y + row] = a[col * y + row] - b;
    }
}