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



// creates a sub-mat of S with indices in e base 0
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

void   makeSdAndM(Eigen::MatrixXd S, Eigen::VectorXi e, Eigen::MatrixXd& M, Eigen::MatrixXd& N,
                  int n, int d){
  // M(d, r) N(d, d)
  if (d >= n) {
    Rf_error("Too many indices to eliminate.\n");
  }
  if (e.maxCoeff() > n){
    Rf_error("largest index greater than the number of columns.\n");
  }
  
  //  Eigen::MatrixXd M(d, r);  
  for (int i = 0; i < d; ++i){
    M.row(i) = S.row(e(i));
  }  
  //  Eigen::MatrixXd N = M.topLeftCorner(d, d);  
  for (int i = 0; i < d; ++i){
    N.col(i) = M.col(e(i));
  }
} 

// Deflates S (pass already deflated and vector current loads)
// returns vexp by ref
void deflSC(Eigen::VectorXd a, Eigen::MatrixXd& K, Eigen::VectorXi ind, double& vexp){
  // # pass only a nonzero loads
  // K = deflated matrix
  // #  K <-- (S - Saa'S/(a'Sa) // deflated S matrix
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
    Rf_error("defSC: tt is not > 0");
  
  // O = Sa/(tt)
  const Eigen::VectorXd O = (t.array()*tt).matrix();
  
  const double cvk = K.trace();
  // K = S - Saa'S/(a'Sa) deflated S
  Eigen::MatrixXd L = t * O.transpose();
  K = K - t * O.transpose(); //deflated S
  vexp =  cvk - K.trace() ;
  
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

// indnot could be used for extracting the indices later, so use -2, -1 and {0:(p-1)}
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

// This is the main function for R
// S correl matrix
// pvexpfs is proportion of PC to explain by each block
// pvexp is proportion total variance of matrix to explain to terminate computing comps
// ncomps nistead of pvexp maximum number of comps (priority)
// full rank small eps to discard vars from selection
// simply projects current full rank PC onto set of variables in ind
// it does not compute the LS SPCA components, it seems to work as well as that
// uses power method to compute PCs
// new version reduces K and S in one function (+6% ) < check cost of resizing
// [[Rcpp::export]]
List fspcaCpmNoe(Eigen::MatrixXd S, double pvexpfs = 0.95, double pvexp = 0.95, 
               int ncomps = 0, double fullrank = 0,  double eps = 10E-5){
  int p = S.cols();
  if (ncomps == 0)
    ncomps = p;
  Eigen::MatrixXd K(S);

  double totvexp = S.trace();// total variance S
  double maxvexp;// this is vexp by first PC for fow_select
  
  Eigen::VectorXd si(p); 

  si = eigvecPMC(S, maxvexp);
//Rcout << "done si first " << endl;
  // here could compute D as   vec * diag(val^2) * vec.transpose 
  
  Eigen::MatrixXd Sinv(p, p);
  Eigen::MatrixXd M(p, p);
  
  
  Eigen::VectorXd a(p);
  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(p, ncomps);
  //  List load(p);
  List indout(p);
  
  Eigen::VectorXd vexp = Eigen::VectorXd::Zero(ncomps);
  Eigen::VectorXd cvexp = vexp;
  double cvt;
  Eigen::VectorXi indj(p);//this to pass to fwd_select 
  
  Eigen::MatrixXd Sd(p, p);// maybe better leave dynamic? this takes S[onlyind, onlyind]

  int cardt = 0;
  Eigen::VectorXi card(p); 
  int nc = 0;   
  bool stopComp = false;
  
  int j = 0;
  while (stopComp == false){
    fwd_selectC(S, indj, cardt, si, maxvexp, pvexpfs, fullrank);
//Rcout << "comp "<< j + 1 << "done fwd_swlwct " << endl;
    
    card(j) = cardt;
    std::sort(indj.data(),indj.data() + cardt);

    // if ( j == 2)
    //   Rf_error("done 1");  
    

    // need make one function that does bot Sd and M
    // create submatrices for computing loaidngs 
    
    Sd.resize(cardt, cardt);
    M.resize(cardt, NoChange);
    makeSdAndM(S, indj.head(cardt), M, Sd, p,cardt);
//    Sd.topLeftCorner(cardt, cardt) = makeSubS(S, indj.head(cardt));
//    M.topLeftCorner(cardt, p) = selectRowsC(K, indj.head(cardt), p);
//Rcout << "comp "<< j + 1 << "done selectRows" << endl;    

    //  compute loadings        
  //  Sinv.topLeftCorner(cardt, cardt)  = Sd.topLeftCorner(cardt, cardt).llt().solve(MatrixXd::Identity(cardt, cardt));
    Sinv.topLeftCorner(cardt, cardt)  = Sd.llt().solve(MatrixXd::Identity(cardt, cardt));
    //Rcout << "comp "<< j + 1 << "done Sinv" << endl;    
    
    // save loadings 
    a.head(cardt) = ((Sinv.topLeftCorner(cardt, cardt) * M * si).array() / maxvexp).matrix();
//Rcout << "comp "<< j + 1 << "done loadings" << endl;    
    // save loadings in column j
    for (int i = 0; i < cardt; i++){
      A(indj(i), j) = a(i);
    }
    // save loadings in list
    indout[j] = indj.head(cardt).array() + 1;

    nc = nc + 1;
    
    
    // this new func deflates S and D using only last vector of loads
    // returns deflated matr by references and vexp (not cum vexp)
    deflSC(a.head(cardt), K, indj.head(cardt), cvt);
//Rcout << "comp "<< j + 1 << "done deflSC" << endl;    
    
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
      // this power method, returns si and passes maxvexp byref
      si = eigvecPMC(K, maxvexp);
//Rcout << "comp "<< j + 1 << "done si" << endl;    
      
      j = j + 1;
    }
  }//end compute comps
  
  IntegerVector idx = Rcpp::seq(0, nc - 1);

  return  List::create(Named("loadings") = A.topLeftCorner(p,nc), Named("ncomps") = nc, 
                       Named("ind") = indout[idx], Named("card") = card.head(nc), 
                       Named("vexp") = vexp.head(nc), Named("cvexp") = cvexp.head(nc));
} 