//////////////////////////////////////////////////////////////////////////
////Dartmouth CS70.01 sample code
////http://www.dartmouth.edu/~boolzhu/cosc70.01.html
////Linear algebra and vector math library
//////////////////////////////////////////////////////////////////////////

#ifndef __MyMatrix_h__
#define __MyMatrix_h__
#include <vector>
#include <iostream>
#include <cassert>
#include <cmath>
#include <fstream>
using namespace std;

class Matrix
{
public:
    int m;                        ////number of rows
    int n;                        ////number of columns
    vector<double> data;        ////element values, we use double for the data type

    ////matrix constructor
    Matrix(const int _m=1,const int _n=1)
    {
        Resize(_m,_n);
    }

    void Resize(const int _m,const int _n)
    {
        m=_m;
        n=_n;
        data.resize(m*n);
        for(int i=0;i<m*n;i++){
            data[i]=0.;
        }
    }

    ////A=B
    void operator = (const Matrix& B)
    {
        Resize(B.m,B.n);

        for(int i=0;i<(int)data.size();i++){
            data[i]=B.data[i];
        }
    }

    ////A={1.,4.,2.,...}, assigning a std::vector to A. A should initialized beforehand
    void operator = (const vector<double>& input_data)
    {
        assert(input_data.size()<=data.size());

        for(int i=0;i<(int)input_data.size();i++){
            data[i]=input_data[i];
        }
    }
  
    ////return whether A==B
       bool operator == (const Matrix& B)
       {
           assert(m == B.m&&n == B.n);
           for (int i = 0; i < (int)data.size(); i++)
           {
               if (fabs(data[i] - B.data[i]) > 1e-6) return false;
           }
           return true;
       }

    ////return -A
    Matrix operator - ()
    {
        Matrix C(m,n);
        for(int i=0;i<(int)data.size();i++){
            C.data[i]=-data[i];
        }
        return C;
    }

    ////random access of a matrix element
    double& operator() (int i,int j)
    {
        assert(i>=0&&i<m&&j>=0&&j<n);
        return data[i*n+j];
    }

    const double& operator() (int i,int j) const
    {
        assert(i>=0&&i<m&&j>=0&&j<n);
        return data[i*n+j];
    }

    ////display matrix in terminal
    friend ostream & operator << (ostream &out,const Matrix &mtx)
    {
        for(int i=0;i<mtx.m;i++){
            for(int j=0;j<mtx.n;j++){
                out<<mtx(i,j)<<", ";
            }
            out<<std::endl;
        }
        return out;
    }

    //////////////////////////////////////////////////////////////////////////
    ////overloaded operators

    ////matrix-matrix additions
    ////return C = A + B
    ////Notice: I use A to refer to the object itself in all my comments,
    ////if you want to self-access in the C++ code, you should use (*this),
    ////e.g., return (*this); means returning the object itself
    ////the comment A+=B; means (*this)+=B; in the code
    
    Matrix operator + (const Matrix& B)
    {
        assert(m==B.m&&n==B.n);

        Matrix C(m,n);
        for(int i=0;i<(int)data.size();i++){
            C.data[i]=data[i]+B.data[i];
        }
        return C;
    }

    ////A+=B
    void operator += (const Matrix& B)
    {
        assert(m==B.m&&n==B.n);

        for(int i=0;i<(int)data.size();i++){
            data[i]+=B.data[i];
        }
    }

    //////////////////////////////////////////////////////////////////////////
    ////Your implementation starts

    ////Task 1: Mimic the "+" and "+=" operators,
    ////implement four new operators: "-", "-=", matrix-scalar multiplications "*" and "*="
    
    ////return A-B
    /*Your function implementation*/
    Matrix operator - (const Matrix& B)
    {
        assert(m==B.m&&n==B.n);
        Matrix C(m,n);
        for (int i=0; i< (int)data.size();i++){
            C.data[i] = data[i] - B.data[i];
        }
        return C;
    }

    ////A=A-B
    /*Your function implementation*/
    void operator -= (const Matrix& B)
    {
        assert(m==B.m && n==B.n);
        for (int i=0; i< (int)data.size();i++){
            data[i] -= B.data[i];
        }
    }

    ////return A*s, with s as a scalar
    /*Your function implementation*/
    Matrix operator * (const double& s)
    {
        Matrix C(m,n);
        for (int i =0; i< (int)data.size(); i++){
            C.data[i] = s*data[i];
        }
        return C;
    }

    

    ////A=A*s, with s as a scalar
    /*Your function implementation*/
    void operator *= (const double& s)
    {
        for (int i =0; i<(int)data.size(); i++){
            data[i] *= s;
        }
    }

    
    Matrix slow_multiplication(const Matrix& B){
        assert(n==B.m);
        //        !(std::cerr << "Incorrect input, try again --> ");
               
                
                Matrix C(m,B.n);
                int i, j, k;
                for (i=0; i<m; i++){
                    for (j=0; j< B.n; j++){
                        for (k=0; k<n; k++){
                            (C)(i,j) += (*this)(i,k)*(B)(k,j);
                        }
                    }
                }
                return C;
    }
    ////Task 2: matrix-matrix multiplication
    ////Hints: there are four steps:
    ////1, check compatibility by an assert;
    ////2, allocate a matrix C with proper size;
    ////3, calculate each element in C by a (left)row-(right)column multiplication
    //// when accessing an element (i,j) in the object itself, use (*this)(i,j)
    ////4, return c
    /*Your function implementation*/
    Matrix operator * (const Matrix& B)
    {
        assert(n==B.m);
//        !(std::cerr << "Incorrect input, try again --> ");
       
        Matrix C(m,B.n);
        
        int i, j, k;
        for (i=0; i<m; i++){
            for (k=0; k<n; k++){
                for (j=0; j< B.n; j++){
                    (C)(i,j) += (*this)(i,k)*(B)(k,j);
                }
            }
        }
        return C;
    }

    ////return an identity matrix
    Matrix Identity()
    {
        assert(m==n);
        for (int i=0; i<m; i++){
            for (int j=0; j<n; j++){
                (*this)(i,j)= 0;
                if (i ==j){
                    (*this)(i,j)= 1;
                }
            }
        }
        return (*this);
    }

    ////return A^T
    /*Your function implementation*/
    // Changes original matrix
    void Transpose()
    {
        Matrix A(n,m);
        for (int i=0; i<m; i++){
            for (int j=0; j<n; j++){
                A(j,i) = (*this)(i,j);
            }
        }
        (*this) = A;
    }
    
    ////return A^T
     /*Your function implementation*/
    // Does not change original matrix
     Matrix T()
     {
         Matrix A(n,m);
         for (int i=0; i<m; i++){
             for (int j=0; j<n; j++){
                 A(j,i) = (*this)(i,j);
             }
         }
         return A;
     }

    ////return a submatrix block A_ijab,
    ////with i,j as the starting element and a,b as the last element
    /*Your function implementation*/
    Matrix Block(const int i,const int j,const int a,const int b)
    {
        assert(m>=i+a && n>=j+b);
        
        Matrix A(a,b);
        for (int p=0; p<a; p++){
            for (int q=0; q<b; q++){
                (A)(p,q) = (*this)(i+p,j+q);
        }
    }
        return A;
    }
    
    // returns a submatrix block
    // with i as the first row and p as the last row
    Matrix Slice(const int i, const int p)
    {
        assert(m>p);
        Matrix A(p-i+1,n);
//        cout<<"i: "<<i<<", q: "<<q<<std::endl;
        for (int a=0; a<=(p-i);a++){
            for(int b=0;b<n;b++)A.data[a*n+b] = data[i*n+a*n+b];
        }
        return A;
            
    }
    Matrix Inverse(){
        Matrix inv(m,n);
        Matrix ident = (*this).Identity();
        
        return inv;
    }
    Matrix Scalar_Identity(const int row, const int collumn, const int scalar)
    {
        assert(m==n);
        Matrix A(m,n);
        for (int i=0; i<m; i++){
            for (int j=0; j<n; j++){
                    (A)(i,j) = 0;
                    if (i ==j){
                           (A)(i,j) =1;
                       }
                    if (j== collumn-1 && i==row-1){
                        (A)(i,j) = scalar;
                    }
                    
                }
        }
        
        return A;
    }
        
        ////LU Decomposition
        /*Your function implementation*/
        void LUDecomposition(Matrix& L, Matrix& U)
        {
            
        }
      Matrix Swap(const int row1, const int row2){

         // swap elements in matrix
         for (int a=0; a< n; a++){
             double minA = (*this)(row1,a);
             double maxA = (*this)(row2,a);
             data[row1*n+a] = maxA;
             data[row2*n+a] = minA;

         }
          return (*this);
      }
        ///This is a member function to solve Ax=b, you only need to pass b as the argument
        Matrix Backward_Substitution(const Matrix& b)
        {
            Matrix x(b.m);
            /*Your implementation starts to solve Ax=b*/
            /*Your implementation ends*/
            x(n-1,0) = b(n-1,0)/(*this)(n-1,n-1);
            
            for (int i=n-1; i>=0; i--){
                double s = b(i,0);
                for (int j=n-1; j>i; j--){
                    s = s-(*this)(i,j)*x(j,0);
                }
                x(i,0)=s/(*this)(i,i);
            }


            return x;
        }
    void Random_Matrix_Generator()
    {
        for (int i=0; i<m; i++){
            for (int j=0;j<n;j++){
                (*this)(i,j) = rand()%20;
            }
        }
    }
    void Noise_Matrix_Generator(double denominator)
    {
        for (int i=0; i<m; i++){
            for (int j=0;j<n;j++){
                (*this)(i,j) =  ((double)rand()/(double)RAND_MAX)/denominator;
            }
        }
    }

    
    void Gram_Schmidt(Matrix& Q, Matrix& R) {
    
        // breaking A into separate column vectors
        vector<Matrix> a_column;
        for (int k=0; k<n; k++) {
            a_column.push_back((*this).Block(0, k, m, 1));
        }

        R.Resize(n,n);
        // breaking Q into separate column vectors
        vector<Matrix> Q_columns;
        for (int i=0; i<n; i++) {
            Q_columns.push_back(a_column[i]); // Q1 = a1
        for (int j=0; j<i; j++) {
            Matrix tranpose_a;
            tranpose_a = a_column[i].T();

            // orthogonalize
            Q_columns[i] -= Q_columns[j]*(tranpose_a*Q_columns[j])(0,0);
            R(j,i) = (tranpose_a*Q_columns[j])(0,0);
            }

        // find magniture of Q[i]
        double magnitude = 0;
        for (int size=0; size<Q_columns[i].m; size++) {
            magnitude +=pow(Q_columns[i](size,0), 2);
        }

        // make sure bectore is linearly independent
        if (magnitude == 0) {cout<<"Not linearly independent"<<endl;}
        assert(magnitude!=0);

        R(i,i) = sqrt(magnitude);
        Q_columns[i] = Q_columns[i]*(1/sqrt(magnitude));// find qhat
            
  
        }
        // Build the Q Matrix
        for (int i=0; i<Q_columns.size(); i++) {
            Matrix current = Q_columns[i];
            for (int j=0; j<current.m; j++) Q(j, i) = current(j, 0);
          }
    }

        Matrix QRFactorization(Matrix& Q, Matrix& R, Matrix& b) {
            // Step 2: solve Rx=y using backward substitution.
            Matrix y(Q.n, Q.m);
            y = Q.T()*b;
            return R.Backward_Substitution(y);
        }
            
        };


        //////////////////////////////////////////////////////////////////////////
        ////Assignment 2: Solve Linear Equations using Gaussian Elimination
        ////Solve linear equations Ax=b, update your results in x; you don't need to return x
        Matrix Solve(Matrix& A, Matrix& b,Matrix& x)
        {
                ////Forward Elimination: transform A into an upper triangular matrix
                //For each row:
                //1. Search for the maximum element in current column
                //2. Swap maximum row with current row
                //3. Make all of the element 0 below current row in current column
          
            // Starting pivot
            for (int k=0; k< A.n-1; k++){
                if ((A)(k,k) == 0){ //if pivot item is equal to 0
                    for (int m=k+1; m< A.m; m++){
                        if ((A)(m,k)!=0){
                            A.Swap(k, m);
                            b.Swap(k, m);
                            break;}
//                        else if (m-1 == A.m){
//                            cout<<"reached end"<<std::endl;
//                            break;
//                        }
                    }
                }
                for (int i=k+1; i<A.n;i++){ //loop through all rows below pivot row
                    double scalar = (A)(i,k)/A(k,k);
                    for (int j=k; j<A.n; j++){ //loop through all collumns
                        (A)(i,j) = (A)(i,j)- scalar*(A)(k,j);
                    }
                    

                    (b)(i,0) = (b)(i,0) - scalar*(b)(k,0);

                }
            }
    //
                ////Backward Substitution: solve unknowns in a reverse order

                /*Your implementation starts to solve Ax=b*/
            return A.Backward_Substitution(b);
                /*Your implementation ends*/
        }

Matrix Solve_Least_Squares(const Matrix& A, const Matrix& b)
    {
    assert(A.m==b.m);
    // Create Vector x with m values
    Matrix x(A.m,1);
    
    // Created transposed matrix A
    Matrix A_T = A;
    A_T.Transpose();

    // Find Matrix (A^T*A)
    Matrix A_hat = A_T*A;

    //Find Matrix (A^T*b)
    Matrix b_hat = b;
    b_hat = A_T*b_hat;

    cout<<"Using Gaussian"<<std::endl;
    // SOLVE (A^T*A)x = (A^T*b)
    return Solve(A_hat, b_hat, x);
}

double Residual_Calculator(Matrix&A, const Matrix& b, const Matrix& x)
{
    //Equation: ||Ax-b||2 = (Ax-b)^T(Ax-b)
    
    // Derive the matrix of (Ax-b)
    Matrix Ax_b = A*x;
    Ax_b = Ax_b-b;
    
    // Derive the transposed matrix of (Ax-b)
    Matrix Tran_Ax_b = Ax_b.T();
    
    // Calculate ||Ax-b||^2
    Matrix r = Tran_Ax_b*Ax_b;
    return r(0,0);
}


Matrix Richardson_Algorithm(const Matrix&A, const Matrix&b, const int k)
{
    //string filename("Richardson_Values.txt");
    std::ofstream myfile;
    myfile.open("/Users/robertdoherty/Desktop/Richardson_Values.txt");
    
    // Calculate one over the sum of squares for matrix A (i.e. 1/(||A||^2))
    double μ=(double)0;
    for (int i=0;i<A.m; i++){
        for (int j=0;j<A.n;j++){
            μ += pow(A(i,j),2);
        }
    }
    μ = 1/μ;
    
    //Copy Matrix A and create new matrix A_T, which is A transposed
    Matrix A_ = A;
    Matrix A_T = A_.T();
    
    // Create matrix x with first guess of all 0's
    Matrix x(A.n,1);
    
    //Calculate x using non-iterative solve least squares function in order to compare values with iterative solver
    Matrix x_hat = Solve_Least_Squares(A, b);

    // Run Richardson's algorithm for k iterations
    for (int i=0;i<=k;i++){
        x = x- (A_T*μ)*(A_*x-b); // x^(k+1) = x^(k) − μA^T (Ax^(k) − b)
        double r = (double)0;
        for (int a=0; a<x.m;a++){
            r += pow(x(a,0)-x_hat(a,0),2);
            
        }
        myfile<<r<<endl;
    }
    
    myfile.close();
    return x;
}
Matrix Gram_Smidth_Algorithm(Matrix&A, const Matrix&b)
{
    //Break A into column vectors
       vector<Matrix>a;
       for (int i=0; i<A.n;i++){
           a.push_back(A.Block(0,i,A.m,1));
       }
       
       //Gram Schmidth Algorithm --> creating Q and R
       vector<Matrix>q;
       for (int i=0; i<a.size();i++){
           q.push_back(a[i]);
           for (int j=0;j<i;j++){
               q[i] -= q[j]*(a[i].T()*q[j])(0,0); // orthogonalization
           }

           // Find magnitude of Q[i]
           double mag=0;
           for (int m=0; m<q[i].m;m++){
               mag += pow(q[i](m,0),2);
           }
           if (mag==0) cout<<"Not linearly independent!"<<std::endl;
           assert(mag !=0);
           q[i] *= (1/sqrt(mag));  // Add unit vector of q_hat
       }
    Matrix Q(q[0].m, (int)q.size());
    for (int i=0;i<q.size();i++){
        for (int j=0; j<q[i].m;j++){
            Q(j,i) = q[i](j,0);
        }
    }
    return Q;
}
Matrix QR_Factorization(Matrix& A, const Matrix& b)
{
    //Break A into column vectors
    vector<Matrix>a;
    for (int i=0; i<A.n;i++){
        a.push_back(A.Block(0,i,A.m,1));
    }
    
    //Gram Schmidth Algorithm --> creating Q and R
    vector<Matrix>Q;
    Matrix R((int)a.size(),(int)a.size());
    for (int i=0; i<a.size();i++){
        Q.push_back(a[i]);
        for (int j=0;j<i;j++){
            Q[i] -= Q[j]*(a[i].T()*Q[j])(0,0); // orthogonalization
            R(j,i) = (a[i].T()*Q[j])(0,0);
        }
        Matrix zero(Q[i].m);

        // Find magnitude of Q[i]
        double mag=0;
        for (int m=0; m<Q[i].m;m++){
            mag += pow(Q[i](m,0),2);
        }
        if (mag==0) cout<<"Not linearly independent!"<<std::endl;
        assert(mag !=0);
        
        R(i,i) = sqrt(mag);
        Q[i] *= (1/sqrt(mag));  // Add unit vector of q_hat
    }
    // Ax=b, we first factorize A as A=QR, then
    
    // Step 1: solve Qy= b, by y = QTb(recall that we have Q−1=QT for an orthogonal matrix).
    Matrix y((int)Q.size(),1);
    Matrix Q_((int)Q[0].m, (int)Q.size());
    for (int j=0; j<Q.size();j++){
        for (int i=0; i<Q[0].m; i++) Q_(i,j)=Q[j](i,0);
       }
    y = Q_.T()*b;

    
    // Step 2: solve Rx=y using backward substitution.
    return R.Backward_Substitution(y);
}
    

#endif
