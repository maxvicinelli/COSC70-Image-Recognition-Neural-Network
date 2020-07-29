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
    Matrix operator - (const Matrix& B)
    {
        assert(m==B.m && n==B.n);
        
        Matrix C(m,n);
        for (int i =0; i<(int) data.size(); i++) {
            C.data[i] = data[i] - B.data [i];
        }
        return C;
    }

    ////A=A-B
    /*Your function implementation*/
    void operator -= (const Matrix& B)
    {
        assert(m == B.m && n ==B.n);
        
        for(int i=0; i< (int) data.size(); i++) {
            data[i] -= B.data[i];
        }
    }

    ////return A*s, with s as a scalar
    /*Your function implementation*/
    Matrix operator * (const double& s)
    {
        Matrix C(m,n);
        for (int i = 0; i < (int) data.size(); i++) {
            C.data[i] = data[i] * s;
        }
        return C;
    }

    ////A=A*s, with s as a scalar
    /*Your function implementation*/
    void operator *= (const double& s)
    {
        for (int i = 0; i < (int) data.size(); i++) {
            data[i] = data[i] * s;
        }
    }

    ////Task 2: matrix-matrix multiplication
    ////Hints: there are four steps:
    ////1, check compatibility by an assert;
    ////2, allocate a matrix C with proper size;
    ////3, calculate each element in C by a (left)row-(right)column multiplication
    //// when accessing an element (i,j) in the object itself, use (*this)(i,j)
    ////4, return c
    /*Your function implementation*/
    Matrix operator * (const Matrix& B){
        assert(n == B.m);
        Matrix C(m, B.n);
    
        for (int i=0; i<m; i++) {
            for (int j=0; j<B.n; j++) {
                int total = 0;
                for (int h=0; h<n; h++) {
                    total += ((*this)(i,h) * B(h,j));
                }
                C(i,j) = total;
            }
        }
        return C;
    }

    ////Task 3: identity, transpose(), block

    ////return an identity matrix
    /*Your function implementation*/
    Matrix Identity() {
        Matrix C(min(m,n),min(m,n));
        for (int i =0; i<m; i++) {
            for (int j = 0; j<n; j++) {
                if (i == j) {
                    C(i,j) = 1;
                }
            }
        }
        return C;
    }

    ////return A^T
    // Your function implementation
    Matrix Transpose() {
        Matrix C(n,m);
        for (int i=0; i<m; i++) {
            for (int j=0; j<n; j++) {
                C(j,i) = (*this)(i,j);
            }
        }
        return C;
    }

    ////return a submatrix block A_ijab,
    ////with i,j as the starting element and a,b as the block size
    /*Your function implementation*/
    Matrix Block(const int i,const int j,const int a,const int b)
    {
        assert(a+i<m && j+b<n);
        Matrix C(a,b);
        for (int x = i; x<i+a; x++) {
            for (int y = j; y<j+b; y++) {
                C(x-i, y-j) = (*this)(x,y);
            }
        }
        return C;
    }

    ////Task 4: implement a function or a set of functions that were not specified in class
    //returns determinant of 2x2 matrix
    double Determinant2x2() {
        assert(n == m);
        double determinant = 1 / ( ((*this)(0,0)*(*this)(1,1)) - ((*this)(0,1)*(*this)(1,0))  );
        return determinant;
    }
};

#endif
