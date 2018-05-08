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
#include <iomanip>
#include <string>

using namespace std;
const double pi = 3.14159265358979323846;
const complex<double> I(0.0, 1.0);

class Condensate{
public:
    Condensate(int ngrid);
    double dr;
    VectorXd r;
    VectorXd phiA;
    int number;
    Condensate(const Condensate& condensate);
};
typedef Matrix<MatrixXd, Dynamic, 1> DistXd;
typedef Matrix<MatrixXcd, Dynamic, 1> DistXcd;

class Ensemble{
public:
    Ensemble(const Condensate& condensate, int lmax, int kgrid, double kmin, double kmax);
    double dr;
    VectorXd r;
    VectorXcd phiA;
    int number;
    double dk;
    VectorXd k;
    DistXd GN;
    DistXcd GA;
    Ensemble(const Ensemble& ensemble);
};

Condensate::Condensate(int ngrid){
    r = VectorXd::LinSpaced(ngrid, 0.0, 3.0 - 3.0/ngrid);
    dr = r(1) - r(0);
    phiA = (VectorXd::Ones(ngrid)*2 - r);
    number = 510000;
}

Condensate::Condensate(const Condensate& condensate){
    dr = condensate.dr;
    r = condensate.r;
    phiA = condensate.phiA;
    number = condensate.number;
}

Ensemble::Ensemble(const Condensate& condensate, int lmax, int kgrid, double kmin, double kmax){
    dr = condensate.dr;
    r = condensate.r;
    phiA = VectorXcd::Zero(condensate.phiA.size());
    phiA.real() = condensate.phiA;
    number = condensate.number;
    k = VectorXd::LinSpaced(kgrid, kmin, kmax);
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
    number = ensemble.number;
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
    yout(n-1) = (phiA(n-2) - 2.0 * phiA(n-1)) / (dr * dr) - phiA(n-2) / (2 * dr * r(n-1));
    return yout;
}

VectorXd box(const VectorXd& r){
    VectorXd V = VectorXd::Zero(r.size());
    for(int i=0; i<r.size(); i++)
	V(i) = 0.5 * (tanh((r(i) - 1.2) * 2) + 1.);
    return V;
}

template <class T>
T Hamiltonian(const T& phiA, const VectorXd& r, double dr, double g, double number, double trap){
    int n = r.size();
    T yout = (-0.5)*laplacian(phiA, r, dr);
    g = g * number;
    yout += g * phiA.cwiseAbs2().cwiseProduct(phiA);
    yout += trap * box(r).cwiseProduct(phiA);
    return yout;
}

void Descent(Condensate& condensate, double dt, double g, double trap){
    int n = condensate.phiA.size();
    Condensate k1(condensate);
    k1.phiA -= Hamiltonian(condensate.phiA, condensate.r, condensate.dr, g, condensate.number, trap) * dt/2;
    Condensate k2(condensate);
    k2.phiA = Hamiltonian(k1.phiA, k1.r, k1.dr, g, condensate.number, trap) * dt;
    condensate.phiA -= k2.phiA;
}

template <class T>
void normalize(T& phiA, const VectorXd& r, double dr){
    double sum = 2*pi * phiA.cwiseAbs2().cwiseProduct(r).sum();
    phiA /= sqrt(sum * dr);
}

void storeData_c(const VectorXd& data_x, const VectorXcd& data_y, string filename){
    ofstream outFile;
    outFile.open(filename);
    if(outFile.is_open()){
        for(int i=0; i<data_x.size(); i++)
            outFile<<data_x(i)<<"\t"<<abs(data_y(i))<<endl;
        outFile.close();
    }
    else
        cout<<"Fail to open file."<<endl;
}

void storeData_r(const VectorXd& data_x, const VectorXd& data_y, string filename){
    ofstream outFile;
    outFile.open(filename);
    if(outFile.is_open()){
        for(int i=0; i<data_x.size(); i++)
            outFile<<data_x(i)<<"\t"<<setprecision(10)<<data_y(i)<<endl;
        outFile.close();
    }
    else
        cout<<"Fail to open file."<<endl;
}


double Alpha(double T_matrix, const VectorXd& k, double dk, double mu){
    int n = k.size();
    VectorXd inverse_propagator = 2*mu*VectorXd::Ones(n) - k.cwiseProduct(k);
    double alpha = -(VectorXd::Ones(n).cwiseQuotient(inverse_propagator)).dot(k)*dk/(2*pi);
    return alpha;
}

MatrixXd evolveGN_rhs(Ensemble& ensemble, int l, double g){
    int n = ensemble.GN(l).rows();
    int m = ensemble.GN(l).cols();
    MatrixXd yout = MatrixXd::Zero(n, m);
    int left = (l? l-1 : 1);
    int even = (l%2 ? 0 : 1);
    if (l==0){
	    yout.row(0) = -1.0 * (ensemble.GN(left).row(1) + ensemble.GN(l+1).row(1)) * ensemble.k.asDiagonal() / ensemble.dr;
    }
    yout.block(1, 0, n-2, m) = -0.5 * (ensemble.GN(left).block(2, 0, n-2, m) - ensemble.GN(left).block(0, 0, n-2, m) + ensemble.GN(l+1).block(2, 0, n-2, m) - ensemble.GN(l+1).block(0, 0, n-2, m)) * ensemble.k.asDiagonal() / (2 * ensemble.dr);
    yout.row(n-1) = -0.5 * (-ensemble.GN(left).row(n-2) - ensemble.GN(l+1).row(n-2)) * ensemble.k.asDiagonal() / (2 * ensemble.dr);
    yout.block(1, 0, n-1, m) += 0.5 * ensemble.r.segment(1, n-1).cwiseInverse().asDiagonal() * ((l-1)*ensemble.GN(left).block(1, 0, n-1, m) - (l+1)*ensemble.GN(l+1).block(1, 0, n-1, m)) * ensemble.k.asDiagonal();
    VectorXcd Delta = VectorXcd::Zero(n);
    VectorXcd GA0 = ensemble.GA(0) * ensemble.k * ensemble.dk / (2*pi);
    Delta = g * (ensemble.number * ensemble.phiA.cwiseProduct(ensemble.phiA) + GA0);
    if (even)
        yout += 2 * (Delta.asDiagonal() * ensemble.GA(l).conjugate()).imag();
    
    return yout;
}

MatrixXcd evolveGA_rhs(Ensemble& ensemble, int l, double g, double mu, double trap){
    int n = ensemble.GA(l).rows();
    int m = ensemble.GA(l).cols();
    MatrixXcd yout = MatrixXd::Zero(n, m);
    if (l==0)
        yout.row(0) = ensemble.GA(l).row(0) * ensemble.k.asDiagonal() * ensemble.k.asDiagonal() - 2 * (ensemble.GA(l).row(1) - ensemble.GA(l).row(0)) * (2) / (ensemble.dr * ensemble.dr) / 4;
    else
	    yout.row(0) = ensemble.GA(l).row(0) * ensemble.k.asDiagonal() * ensemble.k.asDiagonal();
     
    yout.block(1, 0, n-2, m) = ensemble.GA(l).block(1, 0, n-2, m) * ensemble.k.asDiagonal() * ensemble.k.asDiagonal() - ((ensemble.GA(l).block(2, 0, n-2, m) + ensemble.GA(l).block(0, 0, n-2, m) - 2 * ensemble.GA(l).block(1, 0, n-2, m)) / (ensemble.dr * ensemble.dr) + ensemble.r.segment(1, n-2).cwiseInverse().asDiagonal() * (ensemble.GA(l).block(2, 0, n-2, m) - ensemble.GA(l).block(0, 0, n-2, m)) / (2 * ensemble.dr) - l*l * (ensemble.r.segment(1, n-2).cwiseAbs2()).cwiseInverse().asDiagonal() * ensemble.GA(l).block(1, 0, n-2, m))/4;
    yout.row(n-1) = ensemble.GA(l).row(n-1) * ensemble.k.asDiagonal() * ensemble.k.asDiagonal() - ((ensemble.GA(l).row(n-2) - 2 * ensemble.GA(l).row(n-1)) / (ensemble.dr * ensemble.dr) - ensemble.GA(l).row(n-2) / (2 * ensemble.dr * ensemble.r(n-1)) - l*l *  ensemble.GA(l).row(n-1) / (ensemble.r(n-1)*ensemble.r(n-1)))/4;
    VectorXcd Delta = VectorXcd::Zero(n);
    VectorXd GN0 = ensemble.GN(0) * ensemble.k * ensemble.dk / (2*pi);
    VectorXcd GA0 = ensemble.GA(0) * ensemble.k * ensemble.dk / (2*pi);
    Delta = g * (ensemble.number * ensemble.phiA.cwiseProduct(ensemble.phiA) + GA0);
    yout += 4 * g * (ensemble.number * ensemble.phiA.cwiseAbs2() + GN0).asDiagonal() * ensemble.GA(l);
    yout += 2 * trap * box(ensemble.r).asDiagonal() * ensemble.GA(l);
    int even = (l%2 ? 0 : 1);
    int kronecker = !l;
    yout +=  kronecker * Delta * VectorXcd::Ones(m).transpose() + Delta.asDiagonal() * even * 2 * ensemble.GN(l);
    return -I * yout;
}

VectorXcd NormalF(Ensemble& ensemble, double g){
    return 2 * g * (ensemble.GN(0) * ensemble.k).cwiseProduct(ensemble.phiA) * ensemble.dk / (2*pi);
}

VectorXcd AnomalousF(Ensemble& ensemble, double g){
    return g * (ensemble.GA(0) * ensemble.k).cwiseProduct(ensemble.phiA.conjugate()) * ensemble.dk / (2*pi);
}

void Initialize_GA(Ensemble& ensemble, double T_matrix, double mu, double trap){
    int m = ensemble.k.size();
    int n = ensemble.r.size();
    VectorXd propagator = 2*mu*VectorXd::Ones(m) - ensemble.k.cwiseProduct(ensemble.k);
    MatrixXd de = VectorXd::Ones(n) * propagator.transpose() - (4*T_matrix/(1-Alpha(T_matrix, ensemble.k, ensemble.dk, mu)*T_matrix)*ensemble.number*ensemble.phiA.cwiseAbs2() + 2*trap*box(ensemble.r)) * VectorXd::Ones(m).transpose();
    de = MatrixXd::Ones(n, m).cwiseQuotient(de);
    ensemble.GA(0) = T_matrix * ensemble.number * ensemble.phiA.cwiseAbs2().asDiagonal() * de;
}

void evolve(Ensemble& ensemble, double dt, double *scatter, double t, double mu, double T_matrix, double trap){
    int n = ensemble.phiA.size();
    int even = 1;
    double g = (scatter[0] + scatter[1] * sin(scatter[2]*t))*4*pi;
    VectorXd ga = -Alpha(g, ensemble.k, ensemble.dk, mu) * g * ensemble.phiA.cwiseAbs2();
    g /= (1 - Alpha(g, ensemble.k, ensemble.dk, mu) * g);
    Ensemble k1(ensemble);
    k1.phiA -= I * (Hamiltonian(ensemble.phiA, ensemble.r, ensemble.dr, g, ensemble.number, trap) + NormalF(ensemble, g) + AnomalousF(ensemble, g)) * dt/2;
    for(int l=0; l<ensemble.GN.size()-1; l++){
      even = (l%2? 0 : 1);
        k1.GN(l) += evolveGN_rhs(ensemble, l, g) * dt/2;
	if (even)
        k1.GA(l) += evolveGA_rhs(ensemble, l, g, mu, trap) * dt/2;
    }
    Ensemble k2(ensemble);
    g = (scatter[0] + scatter[1] * sin(scatter[2]*(t + dt/2)))*4*pi;
    g /= (1 - Alpha(g, ensemble.k, ensemble.dk, mu) * g);
    ga = -Alpha(g, ensemble.k, ensemble.dk, mu) * g * k1.phiA.cwiseAbs2();
    k2.phiA = I * (Hamiltonian(k1.phiA, k1.r, k1.dr, g, ensemble.number, trap) + NormalF(k1, g) + AnomalousF(k1, g)) * dt;
    ensemble.phiA -= k2.phiA;
    for(int l=0; l<ensemble.GN.size()-1; l++){
        even = (l%2? 0 : 1);
        k2.GN(l) = evolveGN_rhs(k1, l, g) * dt;
        ensemble.GN(l) += k2.GN(l);
        if (even){
            k2.GA(l) = evolveGA_rhs(k1, l, g, mu, trap) * dt;
            ensemble.GA(l) += k2.GA(l);
        }
    }
}

double total_number(Ensemble& ensemble){
	double c = (ensemble.number * ensemble.phiA.cwiseAbs2() + ensemble.GN(0) * ensemble.k * ensemble.dk / (2*pi)).dot(ensemble.r) * ensemble.dr * 2*pi;
	return c;
}

VectorXd current(VectorXd field1, VectorXd field2, double dt, VectorXd& r, double dr){
	int n = r.size();
	VectorXd J = VectorXd::Zero(n);
	for (int i=0; i<n; i++)
		J(i) = -((field2.segment(0, i).cwiseProduct(r.segment(0, i)).sum() + field2(i)*r(i)/2) - (field1.segment(0, i).cwiseProduct(r.segment(0, i)).sum() + field1(i)*r(i)/2)) * dr / dt / (r(i)+dr/2);
	return J;
}

int main(){
    Condensate condensate(300);
    normalize(condensate.phiA, condensate.r, condensate.dr);
    double T_matrix = 4*pi * 0.00003;
    double trap = 286.;
    double dt1 = 0.00000001;
    for(int t=0; t<30000000; t++){
        Descent(condensate, dt1, T_matrix, trap);
	    normalize(condensate.phiA, condensate.r, condensate.dr);
    }
    
    double mu = Hamiltonian(condensate.phiA, condensate.r, condensate.dr, T_matrix, condensate.number, trap)(0) / condensate.phiA(0);
    VectorXd mu_r = Hamiltonian(condensate.phiA, condensate.r, condensate.dr, T_matrix, condensate.number, trap).cwiseQuotient(condensate.phiA);
    storeData_r(condensate.r, condensate.phiA, "CondensateDensity0");

    double scatter[3] = {0.00003, 0.0003, 1600}; //a_dc, a_ac, frequency(omega)
    double kmax = 64.;
    double kmin = 24.;
    Ensemble ensemble(condensate, 8, 81, kmin, kmax);
    Initialize_GA(ensemble, T_matrix, mu, trap);
    double V = T_matrix / (1 - Alpha(T_matrix, ensemble.k, ensemble.dk, mu) * T_matrix);
    
    double time = 0.0;
    double dt2 = 0.0000005;
    
    VectorXd GN0 = ensemble.GN(0) * ensemble.k * ensemble.dk / (2*pi) / ensemble.number;
    VectorXcd GA0 = ensemble.GA(0) * ensemble.k * ensemble.dk / (2*pi) / ensemble.number;
    storeData_c(ensemble.r, GA0, "AnomalousFluctuation0");
    
    VectorXd Jcurrent;
    VectorXd Ncurrent;
    VectorXcd prev_phiA = ensemble.phiA;
    VectorXd prev_GN0 = GN0;

    for(int t=0; t<112001; t++){
        evolve(ensemble, dt2, scatter, time, mu, T_matrix, trap);
        
        if((t+1)/16000 - t/16000 != 0){
            prev_phiA = ensemble.phiA;
            prev_GN0 = ensemble.GN(0) * ensemble.k * ensemble.dk / (2*pi) / ensemble.number;
        }
        if(t/16000 - (t-1)/16000 != 0){
            GN0 = ensemble.GN(0) * ensemble.k * ensemble.dk / (2*pi) / ensemble.number;
            GA0 = ensemble.GA(0) * ensemble.k * ensemble.dk / (2*pi) / ensemble.number;
            storeData_c(ensemble.r, ensemble.phiA.cwiseAbs2(), "CondensateDensity"+to_string(t));
            storeData_r(ensemble.r, GN0, "NormalDensity"+to_string(t));
            storeData_c(ensemble.r, GA0, "AnomalousFluctuation"+to_string(t));
            Jcurrent = current(prev_phiA.cwiseAbs2(), ensemble.phiA.cwiseAbs2(), dt2, ensemble.r, ensemble.dr);
            storeData_r(ensemble.r, Jcurrent, "J"+to_string(t));
            Ncurrent = current(prev_GN0, GN0, dt2, ensemble.r, ensemble.dr);
            storeData_r(ensemble.r, Ncurrent, "Jn"+to_string(t));
            cout<<total_number(ensemble)<<endl;
            storeData_r(ensemble.k, ensemble.GN(0).row(1).transpose(), "GN0_k"+to_string(t));
            storeData_c(ensemble.k, ensemble.GA(0).row(1).transpose(), "GA0_k"+to_string(t));
            storeData_r(ensemble.k, ensemble.GN(1).row(1).transpose(), "GN1_k"+to_string(t));
            storeData_r(ensemble.k, ensemble.GN(2).row(1).transpose(), "GN2_k"+to_string(t));
            storeData_c(ensemble.k, ensemble.GA(2).row(1).transpose(), "GA2_k"+to_string(t));
            storeData_r(ensemble.k, ensemble.GN(3).row(1).transpose(), "GN3_k"+to_string(t));
            storeData_r(ensemble.k, ensemble.GN(4).row(1).transpose(), "GN4_k"+to_string(t));
            storeData_c(ensemble.k, ensemble.GA(4).row(1).transpose(), "GA4_k"+to_string(t));
            storeData_r(ensemble.k, ensemble.GN(5).row(1).transpose(), "GN5_k"+to_string(t));
            storeData_r(ensemble.k, ensemble.GN(6).row(1).transpose(), "GN6_k"+to_string(t));
            storeData_c(ensemble.k, ensemble.GA(6).row(1).transpose(), "GA6_k"+to_string(t));
        }
        time += dt2;
    }
    
    return 0;
    
}
