#include <iostream>
#include <cstdlib>
#include <vector>
#include <math.h>

using namespace std;
const double pi = 3.14159265358979;

vector<double> *gaussian(double sigma, vector<double> *r){
    int n = r->size();
    vector<double> *function = new vector<double>;
    function->reserve(n);
    for(int i=0; i<n; i++){
        function->push_back(1/sqrt(pi)/sigma*exp(-0.5*pow((*r)[i]/sigma, 2)));
    }
    return function;
}

vector<double> *cosine(double lambda, vector<double> *r){
    int n = r->size();
    vector<double> *function = new vector<double>;
    function->reserve(n);
    for(int i=0; i<n; i++){
        function->push_back(cos((*r)[i]*2*pi/lambda));
    }
    return function;
}

vector<double> *linear(double boundary, vector<double> *r){
    int n = r->size();
    vector<double> *function = new vector<double>;
    function->reserve(n);
    for(int i=0; i<n; i++){
        function->push_back(2*(1 - (*r)[i]/boundary)/boundary);
    }
    return function;
}

vector<double> *coordinates(double *range, double step){
    double n = (*(range+1) - *range)/step;
    vector<double> *r = new vector<double>;
    r->reserve(n);
    for(int i=0; i<n; i++){
        r->push_back(*range + step*i);
    }
    return r;
}

vector<double> *laplacian(vector<double> *function, vector<double> *r, double step){
    int n = function->size();
    vector<double> *laplacian = new vector<double>;
    laplacian->push_back(2*((*function)[1] - (*function)[0])/(step*step));
    for(int i=1; i<n-1; i++){
        laplacian->push_back(((*function)[i+1] - (*function)[i-1])/(2*step*(*r)[i]) + ((*function)[i+1] + (*function)[i-1] - 2*(*function)[i])/(step*step));
    }
    laplacian->push_back(((*function)[n-2] - 2*(*function)[n-1])/(step*step) - (*function)[n-2]/(2*step));
    return laplacian;
}

void evolve(vector<double> *psi, vector<double> *r, double step, double dt){
    int n = psi->size();
    vector<double> *k1;
    k1 = laplacian(psi, r, step);
    vector<double> *temp = new vector<double>;
    for(int i=0; i<n; i++){
        temp->push_back((*psi)[i] + (*k1)[i]*dt/2);
    }
    vector<double> *k2;
    k2 = laplacian(temp, r, step);
    for(int i=0; i<n; i++){
        (*psi)[i] = (*psi)[i] + (*k2)[i]*dt;
    }
    delete k1;
    delete k2;
}

void normalize(vector<double> *psi, vector<double> *r, double step){
    int n = psi->size();
    double sum = 0;
    for(int i=0; i<n; i++){
        sum += 2*pi*(*r)[i]*(*psi)[i]*(*psi)[i];
    }
    for(int i=0; i<n; i++){
        (*psi)[i] /= sqrt(sum*step);
    }
}

int main(){
    vector<double> *psi;
    double range[2] = {0, 1.01};
    double step = 0.01;
    vector<double> *r;
    r = coordinates(range, step);
    psi = gaussian(1.0/sqrt(2.0), r);
    //psi = linear(1.0, r);
    //psi = cosine(4.0, r);
    int n = r->size();
    
    double dt = 0.00001;
    
    for(int t=0; t<100000; t++){
        evolve(psi, r, step, dt);
        normalize(psi, r, step);
    }
    
    for(int i=0; i<r->size(); i++){
        cout<<(*r)[i]<<"\t"<<(*psi)[i]<<endl;
    }
    
    return 0;
    
}