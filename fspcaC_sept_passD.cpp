#include <Rcpp.h>
#include <RcppEigen.h>
#include <Eigen/Dense>
#include <queue>
// #include<Eigen/SparseCore>
using namespace Rcpp;
using namespace Eigen;
using namespace std;


// [[Rcpp::depends(RcppEigen)]]

//
using Eigen::Map;               	// 'maps' rather than copies
using Eigen::Matrix;                  //  matrix generic
using Eigen::MatrixXd;                  // variable size matrix, double precision
using Eigen::VectorXd;                  // variable size vector, double precision
using Eigen::Transpositions;
using Eigen::HouseholderQR;    // Fast scalable QR solver
using Eigen::ColPivHouseholderQR;    // Fast scalable QR solver
using Eigen::FullPivHouseholderQR; // slow full (colsand rows pivoting) 
using Eigen::JacobiSVD;
using Eigen::GeneralizedSelfAdjointEigenSolver;    // one of the eigenvalue solvers
using Eigen::SelfAdjointEigenSolver;    // one of the eigenvalue solvers
using Eigen::LLT;
using Eigen::LDLT;
using Rcpp::List;
using Rcpp::wrap;


// ##########  OK vrsione Sept 04 works

// copied to fspca_sept.cpp

// =========================================================================



// creates a sub-mat of S with indices in e
Eigen::MatrixXd makeSubS(Eigen::MatrixXd S, Eigen::VectorXi e){
  int n = S.cols();
  int r = S.rows();
  int d = e.size();
  if (d >= n) {
    Rf_error("Too many indices to eliminate.\n");
  }
  if (e.maxCoeff() > n){
    Rf_error("largest index greater than the number of columns.\n");
  }
  
  Eigen::MatrixXd M(r, d );  
  for (int i = 0; i < d; ++i){
    M.col(i) = S.col(e(i));
  }
  for (int i = 0; i < d; ++i){
    M.row(i) = M.row(e(i));
  }
  
  return M.topLeftCorner(d, d);
} 

// retruns the rows in e and keeps first c columns
Eigen::MatrixXd selectRowsC(Eigen::MatrixXd A, Eigen::VectorXi e, int c){
  // ATTENZIONE INDICES BASE 0
  // ATTENZIONE e must be sorted e(0) < e(1)
  
  int n = A.cols();
  int r = A.rows();
  int d = e.size();
  if (d >= n) {
    Rf_error("Too many indices to eliminate.\n");
  }
  if (e.maxCoeff() > n){
    Rf_error("largest index greater than the number of columns.\n");
  }
  
  Eigen::MatrixXd M(A.topLeftCorner(r,c));   
  for (int i = 0; i < d; ++i){
    M.row(i) = M.row(e(i));
  }
  
  return M.topLeftCorner(d, c);
} 


// Deflates S and D (pass already deflated and vector current loads)
// returns vexp by ref
void deflSandDC(Eigen::VectorXd a, Eigen::MatrixXd& K, 
                Eigen::MatrixXd& D, Eigen::VectorXi ind, double& vexp){
  // # pass only a nonzero loads
  // S = deflated matrix
  // # D = SS
  // #  K <-- (S - Saa'S/(a'Sa) // deflated S matrix
  // # KK deflated product corr matrix D = KK
  // #   KK = D - Daa'S/(a'Sa) - Saa'D/(a'Sa) + Saa'Daa'S/(a'Sa)^2
  // ## ===
  const int n = ind.size();
  const int p = K.cols();
  
  // t = Sa
  Eigen::VectorXd t = Eigen::VectorXd::Zero(p); 
  for (int i = 0; i < p; i++)
    for(int k = 0; k < n; k++) 
      t(i) += K(i, ind(k)) * a(k ); // only elements in ind
  // tt = a'Sa = t'a
  
  double tt = 0.0; 
  for(int k = 0; k < n; k++)
    tt += a(k) * t(ind(k));
  if (tt > 0)
    tt = 1/tt;
  else
    Rf_error("defSandD: tt is not > 0");
  
  // O = Sa/(tt)
  const Eigen::VectorXd O = (t.array()*tt).matrix();
  
  const double cvk = K.trace();
  // K = S - Saa'S/(a'Sa) deflated S
  Eigen::MatrixXd L = t * O.transpose();
  K = K - t * O.transpose(); //deflated S
  vexp =  cvk - K.trace() ;
  
  // deflate D
  
  // N = aa'S/(tt) = a*t'/(tt) = a*O' (n x p)
  Eigen::MatrixXd N = Eigen::MatrixXd::Zero(n, p); 
  for (int i = 0; i < n; i++)
    for (int j = 0; j < p; j++)
      N(i, j) += a(i) * O(j);  
  
  //M = Daa'S/(a'Sa) = D.transpose() * N; // (p, p)
  Eigen::MatrixXd M = Eigen::MatrixXd::Zero(p,p); 
  for (int i = 0; i < p; i++)
    for (int j = 0; j < p; j++)
      for(int k = 0; k < n; k++) 
        M(i, j) += D(i, ind(k)) * N(k, j);  // (p, p)
  
  //  H = N.transpose() * M (p x p)
  Eigen::MatrixXd H = Eigen::MatrixXd::Zero(p,p); // 
  for (int i = 0; i < p; i++)
    for (int j = 0; j < p; j++)
      for(int k = 0; k < n; k++) 
        H(i, j) += N(k, i) * M(ind(k), j);  
  D = (D.array() - M.array() - M.transpose().array() + H.array()).matrix(); 
  return;
}  
//



// finds max part corr exclude small ss, pdates indnot returns ind
int findmax(Eigen::VectorXi& indnot, Eigen::VectorXd vt){
  
  double p = indnot.size();
  double m = 0.0;
  int ind = 0;
  for (int i = 0; i < p; i++){
    if (indnot(i) == -2){
      if(vt(i) > m){
        m = vt(i);
        ind = i;
      }
    }
  }
  indnot(ind) = ind;
  return ind;  
}

// fixed
void fwd_selectC(Eigen::MatrixXd S, Eigen::VectorXi& ind, int& card,
                 Eigen::VectorXd si, double totvexp, double pvexp,
                 double fullrank = 0.0){ 
  Eigen::VectorXd sik = si;
  int p = S.cols();
  // int induno;
  double tmp; 
  Eigen::VectorXd vexpt(p);
  Eigen::VectorXd cvexpt(p);
  Eigen::VectorXd vt(p);
  Eigen::VectorXi indnot = Eigen::VectorXi::Constant(p, -2);
  Eigen::VectorXd ba(p);
  
  for (int i=0; i < p; i++)
    vt(i) = sik(i) * sik(i) / S(i,i);
  
  ind(0) = findmax(indnot, vt);
  
  vexpt(0) = vt(ind(0));
  cvexpt(0) = vt(ind(0));
  int i = 1;
  bool stopSelect = false;
  // start looping ============================================  
  while (stopSelect == false){
    
    tmp = sik(ind(i - 1))/S(ind(i - 1), ind(i - 1));
    for (int j = 0; j < p; j++){
      if ( indnot(j) == -2){
        sik(j) = sik(j) -  (tmp * S(ind(i-1), j));
      }   
      else{
        sik(j) = 0;
      } 
    }  
    
    ba = (S.col(ind(i-1)).array()/sqrt(S(ind(i-1), ind(i-1)))).matrix();
    S = S - ba * ba.transpose();
    
    for (int j = 0; j < p; j++){
      if ( indnot(j) == -2){
        if (S(j,j)> fullrank)
          vt(j) = sik(j) * sik(j)/S(j,j);
        else{
          indnot(j) = -1;
          vt(j) = 0;
        }
      }
      else{
        vt(j) = 0;
      }
    }
    
    ind(i) = findmax(indnot, vt);
    indnot(ind(i)) = 0;
    
    vexpt(i) =  vt(ind(i));
    cvexpt(i) = cvexpt(i-1) + vexpt(i);
    
    if (cvexpt(i) >= pvexp*totvexp){
      card = i + 1;
      stopSelect = true;
    }
    else{
      i = i + 1;
    }
    //    Rcpp::checkUserInterrupt();
    
  }  
}

/* non serve
// power method computes only first eigvec, about 82 times faster tha eigen!
Eigen::VectorXd eigvecPMC(Eigen::MatrixXd& X, double& val, double eps = 10E-5){
  const int p = X.cols();
  double sqp = sqrt(double(p));
  Eigen::VectorXd v0 = VectorXd::Constant(p, 1.0/sqp);
  Eigen::VectorXd v = VectorXd::Constant(p, 0.0);
  double stp = 1.0;
  int k = 0;
  while (stp > eps){
    v = X * v0;
    val = v.norm();
    v = v.array()/val;
    stp = (v0.array() - v.array()).matrix().norm();
    v0 = v;
    k++;
    if (k > 100){
      Rf_warning("Powermethod: not converged in 100 iterations. Error is", k);
      break;//here should use try-catch  
    }  
  }
  //  Rcout << "k = " << k << "; stp = " << stp << endl;
  return (v.array() * val);  
}

*/

// This is the main function for R
// S correl matrix
// pvexpfs is proportion of PC to explain by each block
// pvexp is proportion total variance of matrix to explain to terminate computing comps
// ncomps nistead of pvexp maximum number of comps (priority)
// full rank small eps to discard vars from selection
// newpc if false uses PCs of S not compute newpc each block, for large mats
// pass D
// [[Rcpp::export]]
List fspcaCD(Eigen::MatrixXd S, Eigen::MatrixXd D, double pvexpfs = 0.95, double pvexp = 0.95, 
               int ncomps = 0, double fullrank = 0, bool newpc = true, double eps = 10E-8){
  int p = S.cols();
  if (ncomps == 0)
    ncomps = p;
  Eigen::MatrixXd K(S);
  Eigen::MatrixXd M = D;
  
  SelfAdjointEigenSolver<Eigen::MatrixXd> es(S);
  // here could compute D as   vec * diag(val^2) * vec.transpose 
  
  Eigen::MatrixXd vec  = es.eigenvectors().rowwise().reverse();
  Eigen::VectorXd vexppc = es.eigenvalues().reverse();
  
  double totvexp = vexppc.sum();// total variance S
  double maxvexp = vexppc(0);// this is vexp by first PC for fow_select
  
  Eigen::VectorXd si = vec.col(0) * vexppc(0);
  
  Eigen::VectorXd a(p);
  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(p, ncomps);
  //  List load(p);
  List indout(p);
  
  Eigen::VectorXd vexp = Eigen::VectorXd::Zero(ncomps);
  Eigen::VectorXd cvexp = vexp;
  double cvt;
  Eigen::VectorXi indj(p);//this to pass to fwd_select 
  
  Eigen::MatrixXd Sd(p, p);// this takes S[onlyind, onlyind]
  Eigen::MatrixXd Dd(p, p);// this takes D[onlyind,onlyind] deflated
  
  int cardt = 0;
  int totcard = 0;
  Eigen::VectorXi card(p); 
  int nc = 0;   
  bool stopComp = false;
  
  int j = 0;
  while (stopComp == false){
    fwd_selectC(S, indj, cardt, si, maxvexp, pvexpfs, fullrank);
    
    card(j) = cardt;
    std::sort(indj.data(),indj.data() + cardt);

    // if ( j == 2)
    //   Rf_error("done 1");  
    
    totcard = totcard + cardt;// a che serve ?
    
    // create submatrices for computing loaidngs    
    Sd.topLeftCorner(cardt, cardt) = makeSubS(S, indj.head(cardt));
    Dd.topLeftCorner(cardt, cardt) = makeSubS(M, indj.head(cardt));
    
    //  compute loadings        
    GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> es(Dd.topLeftCorner(cardt, cardt),
                                                          Sd.topLeftCorner(cardt, cardt));
    // save loadings 
    a.head(cardt) = es.eigenvectors().col(cardt - 1);
    // save loadings in column j
    for (int i = 0; i < cardt; i++){
      A(indj(i), j) = es.eigenvectors()(i, cardt - 1);
    }
    // save loadings in list
    indout[j] = indj.head(cardt).array() + 1;

    nc = nc + 1;
    
    // this new func deflates S and M using only last vector of loads
    // returns deflated matr by references and vexp (not cum vexp)
    deflSandDC(a.head(cardt), K, M, indj.head(cardt), cvt);

    vexp(j) = cvt;
    if (j > 0)
      cvexp(j) = cvt + cvexp(j-1);
    else
      cvexp(j) = cvt;

    // checks if stopComp met
    if ((cvexp(j) > pvexp * totvexp) || ((j + 1) == ncomps)){
      stopComp = true;
      ncomps = nc;
    }
    else{
      if (newpc == true){
        SelfAdjointEigenSolver<Eigen::MatrixXd> es(K);
        // // this is ok because X'K = K'K, so X'Kv = K'Kv = v*lambda_1        
        maxvexp =  es.eigenvalues()(p-1);
        si = es.eigenvectors().col(p-1).array() * maxvexp;
        // this power method, returns si and passes maxvexp byref
        //si = eigvecPMC(K, maxvexp, eps);
      }
      else{// this takes the jth pc
        maxvexp =  vexppc(j); 
        si = vec.col(j).array() * maxvexp;
      }  
      j = j + 1;
    }
  }//end compute comps
  
  IntegerVector idx = Rcpp::seq(0, nc - 1);

  return  List::create(Named("loadings") = A.topLeftCorner(p,nc), Named("ncomps") = nc, 
                       Named("ind") = indout[idx], Named("card") = card.head(nc), 
                       Named("vexp") = vexp.head(nc), Named("cvexp") = cvexp.head(nc));
} 