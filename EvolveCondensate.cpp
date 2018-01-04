#include <Eigen/Core>
#include <Eigen/StdVector>
//#include <Eigen/Eigenvalues>
using namespace Eigen;

#include <iostream>
#include <cstdlib>
#include <vector>
#include <math.h>
#include <fstream>
#include <complex>
#include <cmath>

using namespace std;
const double pi = 3.14159265358979;
const complex<double> I(0.0, 1.0);

class Condensate{
public:
    Condensate(int ngrid);
    double dr;
    VectorXd r;
    VectorXd phiA;
    Condensate(const Condensate& condensate);
};

class Ensemble{
public:
    Ensemble(const Condensate& condensate);
    double dr;
    VectorXd r;
    VectorXcd phiA;
    Ensemble(const Ensemble& ensemble);
};

Condensate::Condensate(int ngrid){
    r = VectorXd::LinSpaced(ngrid, 0.0, 1.0);
    dr = r(1) - r(0);
    phiA = VectorXd::Ones(ngrid) - r;
}

Condensate::Condensate(const Condensate& condensate){
    dr = condensate.dr;
    r = condensate.r;
    phiA = condensate.phiA;
}

Ensemble:: Ensemble(const Condensate& condensate){
    dr = condensate.dr;
    r = condensate.r;
    phiA = VectorXcd::Zero(condensate.phiA.size());
    phiA.real() = condensate.phiA;
}

Ensemble:: Ensemble(const Ensemble& ensemble){
    dr = ensemble.dr;
    r = ensemble.r;
    phiA = ensemble.phiA;
}

template <class T>
T laplacian(const T& phiA, const VectorXd& r, double dr){
    int n = phiA.size();
    T yout = T::Zero(n);
    yout(0) = (4.0 * (phiA(1) - phiA(0)) / (dr * dr));
    yout.segment(1, n-2) = (phiA.segment(2, n-2) - phiA.segment(0, n-2)).cwiseQuotient(2 * dr * r.segment(1, n-2)) + (phiA.segment(2, n-2) + phiA.segment(0, n-2) - 2.0 * phiA.segment(1, n-2)) / (dr * dr);
    yout(n-1) = (phiA(n-2) - 2.0 * phiA(n-1)) / (dr * dr) - phiA(n-2) / (2 * dr);
    return yout;
}

template <class T>
T Hamiltonian(const T& phiA, const VectorXd& r, double dr, double a){
    T yout = (-0.5)*laplacian(phiA, r, dr);
    double potential = 4.0*pi * a; //* number  density 
    yout += potential * phiA.cwiseAbs2().cwiseProduct(phiA);
    return yout;
}

void Descent(Condensate& condensate, double dt, double a){
    int n = condensate.phiA.size();
    Condensate k1(condensate);
    k1.phiA -= Hamiltonian(condensate.phiA, condensate.r, condensate.dr, a) * dt/2;
    Condensate k2(condensate);
    k2.phiA = Hamiltonian(k1.phiA, k1.r, k1.dr, a) * dt;
    condensate.phiA -= k2.phiA;
}

void evolve(Ensemble& ensemble, double dt, double *scatter, double t){
    int n = ensemble.phiA.size();
    double a = scatter[0] + scatter[1] * sin(scatter[2]*t);
    Ensemble k1(ensemble);
    k1.phiA -= I * Hamiltonian(ensemble.phiA, ensemble.r, ensemble.dr, 2*a) * dt/2;
    Ensemble k2(ensemble);
    a = scatter[0] + scatter[1] * sin(scatter[2]*(t + dt/2)); //t or t+dt/2 ?
    k2.phiA = I * Hamiltonian(k1.phiA, k1.r, k1.dr, 2*a) * dt;
    ensemble.phiA -= k2.phiA;
}

template <class T>
void normalize(T& phiA, const VectorXd& r, double dr){
    double sum = 2*pi * phiA.cwiseAbs2().cwiseProduct(r).sum();
    phiA /= sqrt(sum * dr);
}

int main(){
    Condensate condensate(101);
    normalize(condensate.phiA, condensate.r, condensate.dr);
    double scatter[3] = {1.0, 0.0, 1.0}; //a_dc, a_ac, frequency(omega)
    double dt1 = 0.00001;
    for(int t=0; t<100000; t++){
        Descent(condensate, dt1, scatter[0]);
        normalize(condensate.phiA, condensate.r, condensate.dr);
    }
    
    Ensemble ensemble(condensate);
    double time = 0.0;
    double dt2 = 0.00001;
    for(int t=0; t<100000; t++){
        evolve(ensemble, dt2, scatter, time);
        //normalize(ensemble.phiA, ensemble.r, ensemble.dr);
        time += dt2;
    }
    
    ofstream outFile;
    outFile.open("Evolve.txt");
    if(outFile.is_open()){
        for(int i=0; i<condensate.r.size(); i++)
            outFile<<condensate.r(i)<<"\t"<<norm(ensemble.phiA(i))<<endl;
    }
    else
        cout<<"Fail to open file."<<endl;
    
    
    return 0;
    
}