#include <Eigen/Core>
#include <Eigen/StdVector>
using namespace Eigen;

#include <iostream>
#include <cstdlib>
#include <vector>
#include <math.h>
#include <fstream>
#include <complex>
#include <cmath>

using namespace std;
const double pi = 3.14159265358979323846;
const complex<double> I(0.0, 1.0);

class Condensate{
public:
    Condensate(int ngrid);
    double dr;
    VectorXd r;
    VectorXd phiA;
    Condensate(const Condensate& condensate);
};
typedef Matrix<MatrixXd, Dynamic, 1> DistXd;
typedef Matrix<MatrixXcd, Dynamic, 1> DistXcd;

class Ensemble{
public:
    Ensemble(const Condensate& condensate, int lmax, int kgrid);
    double dr;
    VectorXd r;
    VectorXcd phiA;
    double dk;
    VectorXd k;
    DistXd GN;
    DistXcd GA;
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

Ensemble::Ensemble(const Condensate& condensate, int lmax, int kgrid){
    dr = condensate.dr;
    r = condensate.r;
    phiA = VectorXcd::Zero(condensate.phiA.size());
    phiA.real() = condensate.phiA;
    k = VectorXd::LinSpaced(kgrid, 0.0, 2*pi);
    dk = k(1) - k(0);
    GN = DistXd(lmax);
    GA = DistXcd(lmax);
    for(int i=0; i<lmax; i++){
        GN(i) = MatrixXd::Zero(condensate.phiA.size(), kgrid);
        GA(i) = MatrixXcd::Zero(condensate.phiA.size(), kgrid);
    }
}

Ensemble::Ensemble(const Ensemble& ensemble){
    dr = ensemble.dr;
    r = ensemble.r;
    phiA = ensemble.phiA;
    dk = ensemble.dk;
    k = ensemble.k;
    GA = ensemble.GA;
    GN = ensemble.GN;
}

template <class T>
T laplacian(const T& phiA, const VectorXd& r, double dr){
    int n = phiA.size();
    T yout = T::Zero(n);
    yout(0) = 4.0 * (phiA(1) - phiA(0)) / (dr * dr);
    yout.segment(1, n-2) = (phiA.segment(2, n-2) - phiA.segment(0, n-2)).cwiseQuotient(2 * dr * r.segment(1, n-2)) + (phiA.segment(2, n-2) + phiA.segment(0, n-2) - 2.0 * phiA.segment(1, n-2)) / (dr * dr);
    yout(n-1) = (phiA(n-2) - 2.0 * phiA(n-1)) / (dr * dr) - phiA(n-2) / (2 * dr);
    return yout;
}

template <class T>
T Hamiltonian(const T& phiA, const VectorXd& r, double dr, double g){
    T yout = (-0.5)*laplacian(phiA, r, dr);
    yout += g * phiA.cwiseAbs2().cwiseProduct(phiA);
    return yout;
}

void Descent(Condensate& condensate, double dt, double g){
    int n = condensate.phiA.size();
    Condensate k1(condensate);
    k1.phiA -= Hamiltonian(condensate.phiA, condensate.r, condensate.dr, g) * dt/2;
    Condensate k2(condensate);
    k2.phiA = Hamiltonian(k1.phiA, k1.r, k1.dr, g) * dt;
    condensate.phiA -= k2.phiA;
}

template <class T>
void normalize(T& phiA, const VectorXd& r, double dr){
    double sum = 2*pi * phiA.cwiseAbs2().cwiseProduct(r).sum();
    phiA /= sqrt(sum * dr);
}

MatrixXd evolveGN_rhs(Ensemble& ensemble, int l, double g){
    int n = ensemble.GN(l).rows();
    int m = ensemble.GN(l).cols();
    MatrixXd yout = MatrixXd::Zero(n, m);
    int left = (l? l-1 : 1);
    yout.block(1, 0, n-2, m) = -0.5 * (ensemble.GN(left).block(2, 0, n-2, m) - ensemble.GN(left).block(0, 0, n-2, m) + ensemble.GN(l+1).block(2, 0, n-2, m) - ensemble.GN(l+1).block(0, 0, n-2, m)) * ensemble.k.asDiagonal() / (2 * ensemble.dr);
    yout.row(n-1) = -0.5 * (-ensemble.GN(left).row(n-2) - ensemble.GN(l+1).row(n-2)) * ensemble.k.asDiagonal() / (2 * ensemble.dr);
    yout.block(1, 0, n-1, m) += 0.5 * VectorXd::Ones(n-1).cwiseQuotient(ensemble.r.segment(1, n-1)).asDiagonal() * ((l-1)*ensemble.GN(left).block(1, 0, n-1, m) - (l+1)*ensemble.GN(l+1).block(1, 0, n-1, m)) * ensemble.k.asDiagonal();
    VectorXcd Delta = VectorXcd::Zero(n);
    VectorXcd GA0 = ensemble.GA(0) * ensemble.k * ensemble.dk;
    Delta = g * (ensemble.phiA.cwiseProduct(ensemble.phiA) + GA0);
    yout += 2 * (Delta.asDiagonal() * ensemble.GA(l).conjugate()).imag();
    return yout;
}

MatrixXcd evolveGA_rhs(Ensemble& ensemble, int l, double g){
    int n = ensemble.GA(l).rows();
    int m = ensemble.GA(l).cols();
    MatrixXcd yout = MatrixXd::Zero(n, m);
    yout.row(0) = -(2 * (ensemble.GA(l).row(1) - ensemble.GA(l).row(0)) * (2 - l*l/2) / (ensemble.dr * ensemble.dr) - ensemble.GA(l).row(0) * ensemble.k.asDiagonal() * ensemble.k.asDiagonal() / 4);
    yout.block(1, 0, n-2, m) = -((ensemble.GA(l).block(2, 0, n-2, m) + ensemble.GA(l).block(0, 0, n-2, m) - 2 * ensemble.GA(l).block(1, 0, n-2, m)) / (ensemble.dr * ensemble.dr) + VectorXcd::Ones(n-2).cwiseQuotient(ensemble.r.segment(1, n-2)).asDiagonal() * (ensemble.GA(l).block(2, 0, n-2, m) - ensemble.GA(l).block(0, 0, n-2, m)) / (2 * ensemble.dr) - l*l * (VectorXcd::Ones(n-2).cwiseQuotient(ensemble.r.segment(1, n-2)).cwiseQuotient(ensemble.r.segment(1, n-2))).asDiagonal() * ensemble.GA(l).block(1, 0, n-2, m) - ensemble.GA(l).block(1, 0, n-2, m) * ensemble.k.asDiagonal() * ensemble.k.asDiagonal() / 4);
    yout.row(n-1) = -((ensemble.GA(l).row(n-2) - 2 * ensemble.GA(l).row(n-1)) / (ensemble.dr * ensemble.dr) - ensemble.GA(l).row(n-2) / (2 * ensemble.dr) - l*l *  ensemble.GA(l).row(n-1) - ensemble.GA(l).row(n-1) * ensemble.k.asDiagonal() * ensemble.k.asDiagonal() / 4);
    VectorXcd Delta = VectorXcd::Zero(n);
    VectorXcd GN0 = ensemble.GN(0) * ensemble.k * ensemble.dk;
    Delta = g * (ensemble.phiA.cwiseProduct(ensemble.phiA) + GN0);
    yout += 4 * g * (ensemble.phiA.cwiseAbs2() + GN0).asDiagonal() * ensemble.GA(l);
    int kronecker = !l;
    yout += Delta.asDiagonal() * (kronecker * MatrixXcd::Identity(n, m) + 2 * ensemble.GN(l));
    return -I * yout;
}

VectorXcd NormalF(Ensemble& ensemble, double g){
    return 2 * g * (ensemble.GN(0) * ensemble.k).cwiseProduct(ensemble.phiA) * ensemble.dk;
}

VectorXcd AnomalousF(Ensemble& ensemble, double g){
    return 2 * g * (ensemble.GA(0) * ensemble.k).cwiseProduct(ensemble.phiA.conjugate()) * ensemble.dk;
}

void evolve(Ensemble& ensemble, double dt, double *scatter, double t){
    int n = ensemble.phiA.size();
    double g = scatter[0] + scatter[1] * sin(scatter[2]*t);
    Ensemble k1(ensemble);
    k1.phiA -= I * (Hamiltonian(ensemble.phiA, ensemble.r, ensemble.dr, g) + NormalF(ensemble, g) + AnomalousF(ensemble, g)) * dt/2;
    for(int l=0; l<ensemble.GN.size()-1; l++){
        k1.GN(l) += evolveGN_rhs(ensemble, l, g) * dt/2;
        k1.GA(l) += evolveGA_rhs(ensemble, l, g) * dt/2;
    }
    Ensemble k2(ensemble);
    g = scatter[0] + scatter[1] * sin(scatter[2]*(t + dt/2));
    k2.phiA = I * (Hamiltonian(k1.phiA, k1.r, k1.dr, g) + NormalF(k1, g) + AnomalousF(k1, g)) * dt;
    ensemble.phiA -= k2.phiA;
    for(int l=0; l<ensemble.GN.size()-1; l++){
        k2.GN(l) = evolveGN_rhs(k1, l, g) * dt;
        ensemble.GN(l) += k2.GN(l);
        k2.GA(l) = evolveGA_rhs(k1, l, g) * dt;
        ensemble.GA(l) += k2.GA(l);
    }
}

int main(){
    Condensate condensate(51);
    normalize(condensate.phiA, condensate.r, condensate.dr);
    double scatter[3] = {0.00003129, 0.00037354, 1.0}; //a_dc, a_ac, frequency(omega)
    scatter[0] *= 4*pi * 510000;
    scatter[1] *= 4*pi * 510000;
    double dt1 = 0.00001;
    for(int t=0; t<150000; t++){
        Descent(condensate, dt1, scatter[0]);
        normalize(condensate.phiA, condensate.r, condensate.dr);
    }
    
    Ensemble ensemble(condensate, 3, 51);
    double time = 0.0;
    double dt2 = 0.000001;
    for(int t=0; t<30000; t++){
        evolve(ensemble, dt2, scatter, time);
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