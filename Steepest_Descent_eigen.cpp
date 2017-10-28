#include <Eigen/Core>
#include <Eigen/StdVector>
using namespace Eigen;

#include <iostream>
#include <cstdlib>
#include <vector>
#include <math.h>
using namespace std;
const double pi = 3.14159265358979;

class Condensate{
public:
    Condensate(int ngrid);
    double dr;
    VectorXd r;
    VectorXd phiA;
    Condensate(const Condensate& condensate);
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

VectorXd laplacian(const Condensate& condensate){
    int n = condensate.phiA.size();
    VectorXd yout = VectorXd::Zero(n);
    yout(0) = (4 * (condensate.phiA(1) - condensate.phiA(0)) / (condensate.dr * condensate.dr));
    yout.segment(1, n-2) = (condensate.phiA.segment(2, n-2) - condensate.phiA.segment(0, n-2)).cwiseQuotient(2 * condensate.dr * condensate.r.segment(1, n-2)) + (condensate.phiA.segment(2, n-2) + condensate.phiA.segment(0, n-2) - 2 * condensate.phiA.segment(1, n-2)) / (condensate.dr * condensate.dr);
    yout(n-1) = (condensate.phiA(n-2) - 2 * condensate.phiA(n-1)) / (condensate.dr * condensate.dr) - condensate.phiA(n-2) / (2 * condensate.dr);
    return yout;
}

void evolve(Condensate& condensate, double dt){
    int n = condensate.phiA.size();
    Condensate k1(condensate);
    k1.phiA += laplacian(condensate) * dt/2;
    Condensate k2(condensate);
    k2.phiA = laplacian(k1) * dt;
    condensate.phiA += k2.phiA;
}

void normalize(Condensate& condensate){
    double sum = 2*pi * condensate.phiA.cwiseAbs2().cwiseProduct(condensate.r).sum();
    condensate.phiA /= sqrt(sum * condensate.dr);
}

int main(){
    Condensate condensate(101);
    normalize(condensate);
    double dt = 0.00001;
    for(int t=0; t<100000; t++){
        evolve(condensate, dt);
        normalize(condensate);
    }
    cout<<"r"<<endl<<condensate.r<<endl;
    cout<<"phiA"<<endl<<condensate.phiA<<endl;
    
    return 0;
    
}
