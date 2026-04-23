#include "matrixUtil.hpp"
#include <vector>
#include <random>
#include <fstream>
#include <string>
#include <iostream>


std::default_random_engine generator;
std::uniform_real_distribution<double> distribution(-1.0, 1.0);
double generateRandomWeight() {
    return distribution(generator);
}

std::string* risultati = new std::string[2]{"cerchio", "quadrato"};

std::string* humanReadable = new std::string[16]{"quadrato", "quadrato", "quadrato", "quadrato", "quadrato", "quadrato", "quadrato", "cerchio", "cerchio", "cerchio", "cerchio", "cerchio", "cerchio", "cerchio", "cerchio", "quadrato"};
int* cosa = new int[16] {0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0};

const double learning_rate = 3e-4;

double* getImage(int id) {
    int length = 256;
    std::string filename = "immagini/" + std::to_string(id) + ".txt"; 
    std::ifstream in(filename);
    
    double* lista = new double[length];

    for (int i = 0; i < length; ++i) {
        in >> lista[i];
    }
    in.close();
    return lista;
}
double* calcLoss(double target, double response){
    double dif = target - response;
    double segno = dif > 0 ? -1 : 1;
    dif *= dif;
    double* arr = new double[2]{dif, segno};
    return arr;
}

class Layer{
    public:
        Layer(int x1, int y1) : x(x1), y(y1){
            weight = (double*)malloc(sizeof(double) * x1 * y1);
            for(int i = 0; i < x * y; ++i){
                weight[i] = 0.5; //generateRandomWeight();
            }
            InitKernelDimensions();
        }

        void forward(double* c1, double* b1){
            if(x * y > 512){
                ThreadsPerBlock.x = 512;
                ThreadsPerBlock.y = 512;
                blocksPerGrid.x = ceil(x / 512);
                blocksPerGrid.y = ceil(y / 512);
            }
            
            double* c;
            double* b;

            cudaMallocManaged(&c, x * y * sizeof(double));
            cudaMallocManaged(&b, x * y * sizeof(double));

            cudaMemcpy(b, b1, sizeof(double) * x * y, cudaMemcpyHostToDevice);
            cudaMemcpy(c, c1, sizeof(double) * x * y, cudaMemcpyHostToDevice);

            matrixMult<<<blocksPerGrid, ThreadsPerBlock>>>(c, weight, b, x, y, x);
            cudaDeviceSynchronize();
            matrixSigmoid<<<blocksPerGrid, ThreadsPerBlock>>>(b, x, y);
            cudaDeviceSynchronize();

            cudaMemcpy(b1, b, sizeof(double) * x * y, cudaMemcpyHostToHost);
            cudaMemcpy(c1, c, sizeof(double) * x * y, cudaMemcpyHostToHost);

            cudaFree(b);
            cudaFree(c);
        }
        void modifyWeights(double loss, double learning_rate, int segno){
            if(x * y > 512){
                ThreadsPerBlock.x = 512;
                ThreadsPerBlock.y = 512;
                blocksPerGrid.x = ceil(x / 512.0);
                blocksPerGrid.y = ceil(y / 512.0);
            }

            double* weight1;
            double* l;
            cudaMallocManaged(&weight1, x * y * sizeof(double));
            cudaMallocManaged(&l, x * y * sizeof(double));

            cudaMemcpy(weight1, weight, sizeof(double) * x * y, cudaMemcpyHostToDevice);

            for(int i = 0; i < x * y; ++i){
                l[i] = loss * learning_rate * segno;
            }

            matrixDif<<<blocksPerGrid, ThreadsPerBlock>>>(weight1, l, weight1, x, y);
            cudaDeviceSynchronize();

            cudaMemcpy(weight, weight1, sizeof(double) * x * y, cudaMemcpyHostToHost);


            cudaFree(weight1);
            cudaFree(l);
        }
        void modifyWeights(double* loss, double learning_rate, int segno){
            if(x * y > 512){
                ThreadsPerBlock.x = 512;
                ThreadsPerBlock.y = 512;
                blocksPerGrid.x = ceil(x / 512.0);
                blocksPerGrid.y = ceil(y / 512.0);
            }
            double* weight1;
            double* l;

            //std::cout << sizeof(loss) / sizeof(double) << std::endl;
            
            cudaMallocManaged(&weight1, x * y * sizeof(double));
            cudaMallocManaged(&l, x * y * sizeof(double));

            cudaMemcpy(weight1, weight, sizeof(double) * x * y, cudaMemcpyHostToDevice);
            
            for(int i = 0; i < x * y; ++i){
                l[i] = loss[i] * learning_rate;
            }

            //matrixMultInt<<<blocksPerGrid, ThreadsPerBlock>>>(l, segno * learning_rate, l, x, y);
            //cudaDeviceSynchronize();
            matrixDif<<<blocksPerGrid, ThreadsPerBlock>>>(weight1, l, weight1, x, y);
            cudaDeviceSynchronize();

            for(int i = 0; i < x * y; ++i){
                if(weight1[i] != 0.5)
                    std::cout << weight1[i] << "\t";
            }

            cudaMemcpy(weight, weight1, sizeof(double) * x * y, cudaMemcpyHostToHost);

            cudaFree(weight1);
            cudaFree(l);
        }
        void print_weights(){
            for(int i = 0; i < x * y; ++i){
                std::cout << weight[i] << "\t";
                if(i % y == 0)
                    std::cout << "\n";
            }
        }
    private:
        int x;
        int y;
        double* weight;
        dim3 ThreadsPerBlock;
        dim3 blocksPerGrid;

        void InitKernelDimensions(){
            ThreadsPerBlock.x = x * y;
            ThreadsPerBlock.y = x * y;
            blocksPerGrid.x = 1;
            blocksPerGrid.y = 1;
        }
};
class Rete{
    public:
        Rete(int lay, int sx, int sy, int o) : layers(lay), x(sx), y(sy), out(o), layer(layers, Layer(x, y)){
            lastLayer = (double*)malloc(sizeof(double) * y);
            for(int i = 0; i < y; ++i)
                lastLayer[i] = 0.5;
            InitKernelDimensions();
        };

        double* forward(double* a1){
            if(x * y > 512){
                ThreadsPerBlock.x = 512;
                ThreadsPerBlock.y = 512;
                blocksPerGrid.x = ceil(x / 512);
                blocksPerGrid.y = ceil(y / 512);
            }
            for(int i = 0; i < layers; ++i){
                layer[i].forward(a1, a1);
            }
            double* weight1;
            double* a;

            cudaMallocManaged(&weight1, y * sizeof(double));
            cudaMallocManaged(&a, x * y * sizeof(double));

            cudaMemcpy(weight1, lastLayer, sizeof(double) * y, cudaMemcpyHostToDevice);
            cudaMemcpy(a, a1, sizeof(double) * x * y, cudaMemcpyHostToDevice);

            matrixMult<<<blocksPerGrid, ThreadsPerBlock>>>(a, weight1, a, x, x, out);
            cudaDeviceSynchronize();
            matrixTrasponi<<<blocksPerGrid, ThreadsPerBlock>>>(a, a, x, 1);
            cudaDeviceSynchronize();
            matrixMult<<<blocksPerGrid, ThreadsPerBlock>>>(a, weight1, a, x, x, out);
            cudaDeviceSynchronize();
            matrixSigmoid<<<blocksPerGrid, ThreadsPerBlock>>>(a, 1, 1);
            cudaDeviceSynchronize();

            cudaMemcpy(a1, a, sizeof(double) * x * y, cudaMemcpyHostToHost);

            cudaFree(a);
            cudaFree(weight1);
            return a1;
        }
        void backwards(double loss, double learning_rate, int segno){
            if(x * y > 512){
                ThreadsPerBlock.x = 512;
                ThreadsPerBlock.y = 512;
                blocksPerGrid.x = ceil(x / 512);
                blocksPerGrid.y = ceil(y / 512);
            }
            for(int i = 0; i < layers; ++i){
                layer[i].modifyWeights(loss, learning_rate, segno);
            }

            matrixDifInt<<<blocksPerGrid, ThreadsPerBlock>>>(lastLayer, loss * learning_rate, lastLayer, 1, y);
            cudaDeviceSynchronize();
        }
        void backwards(double* loss, double learning_rate, int segno){
            if(x * y > 512){
                ThreadsPerBlock.x = 512;
                ThreadsPerBlock.y = 512;
                blocksPerGrid.x = ceil(x / 512);
                blocksPerGrid.y = ceil(y / 512);
            }
            for(int i = 0; i < layers; ++i){
                layer[i].modifyWeights(loss, learning_rate, segno);
            }
            double* weight1;
            double* l;
            cudaMallocManaged(&weight1, y * sizeof(double));
            cudaMallocManaged(&l, y * sizeof(double));
            
            cudaMemcpy(weight1, lastLayer, sizeof(double) * y, cudaMemcpyHostToDevice);

            for(int i = 0; i < y; ++i){
                double p = 1;
                for(int j = 0; j < x; ++j)
                    p += loss[i * y + j];
                l[i] = p / x * learning_rate;
            }
            matrixDif<<<blocksPerGrid, ThreadsPerBlock>>>(weight1, l, weight1, 1, y);
            cudaDeviceSynchronize();
            //matrixSigmoid<<<blocksPerGrid, ThreadsPerBlock>>>(weight1, 1, y);
            //cudaDeviceSynchronize();

            cudaMemcpy(lastLayer, weight1, sizeof(double) * y, cudaMemcpyHostToHost);

            cudaFree(weight1);
            cudaFree(l);
        }
        void print(){
            for(int i = 0; i < layers; ++i){
                layer[i].print_weights();
            }
            std::cout << std::endl;
            std::cout << std::endl;
            for(int i = 0; i < y; ++i){
                std::cout << lastLayer[i] << "\t";
            }
            std::cout << std::endl;
        }
    private:
        int layers;
        int x;
        int y;
        int out;
        std::vector<Layer> layer;
        double* lastLayer;
        dim3 ThreadsPerBlock;
        dim3 blocksPerGrid;

        void InitKernelDimensions(){
            ThreadsPerBlock.x = x;
            ThreadsPerBlock.y = y;
            blocksPerGrid.x = 1;
            blocksPerGrid.y = 1;
        }
};
void prova(Rete r){
    int j = 0;
    for(int i = 0; i < 16; ++i){
        double* a = getImage(i);
        a = r.forward(a);
        int d = a[0] > 0.75 ? 1 : 0;
        std::cout << i << " risultato " << d << " dando " << a[0] << std::endl;
        double* b = calcLoss(cosa[i], a[0]);

        std::cout << "loss " << b[0] << " segno " << b[1] << std::endl;

        std::cout << "expected " << cosa[i] << std::endl;
        std::cout << std::endl;

        if(d == cosa[i])
            j++;

        delete[] a;
        delete[] b;
    }
    std::cout << "sulle 16 prove ne ha date giuste " << j << std::endl;
}

int main(){
    Rete r = Rete(1, 16, 16, 1);

    prova(r);
    for(int j = 0; j < 16; ++j){
        for(int i = 0; i < 16; ++i){
            double* a = getImage(i);
            double* c = a;
            a = r.forward(a);
            //std::cout << a[0] << "\n";
            double* b = calcLoss(cosa[i], a[0]);
            r.backwards(c, learning_rate, b[1]);

            delete[] a;
            delete[] b;
        }
    }
    r.print();
    prova(r);
    std::cout << cudaGetLastError << std::endl;
}