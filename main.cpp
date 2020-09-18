#include <iostream>
#include <cmath>
#include "MyMatrix.h"
#include "MySparseMatrix.h"
#include "MyGraph.h"
#include "MySparseGraph.h"
#include <chrono>
#include "MyNeuralNetwork.h"
#include "App_tasks.h"
using namespace std::chrono;

void Test_HW1()
{
    ////test the sample code
    Matrix m1(3,3),m2(3,3);
    
    m1={1.,2.,3.,4.,5.,6.,7.,8.,9.};
    m2={1.,0.,0.,0.,2.,0.,0.,0.,3.};
    cout<<"m1:\n"<<m1<<endl;
    cout<<"m2:\n"<<m2<<endl;

    cout<<"m1+m2:\n"<<(m1+m2)<<endl;

    Matrix m3=m1;
    cout<<"m3=m1, m3:\n"<<m3<<endl;

    //////////////////////////////////////////////////////////////////////////
    ////start to test your implementation
    ////Notice the code will not compile before you implement your corresponding operator
    ////so uncomment them one by one if you want to test each function separately

    ////test subtractions
    
    Matrix m4=-m3;
    cout<<"m4=-m3, m4:\n"<<m4<<endl;
    
    m4=m1-m3;
    cout<<"m4=m1-m3,m4:\n"<<m4<<endl;

    m4-=m1;
    cout<<"m4-=m1, m4:\n"<<m4<<endl;
    

    ////test matrix-scalar products
    
    double s=2;
    Matrix m5=m4*s;
    cout<<"m5=m4*s, m5:\n"<<m5<<endl;
    m5*=s;
    cout<<"m5*=s, m5:\n"<<m5<<endl;
    

    ////check matrix-matrix multiplication
    
    Matrix v1(3,1);    ////column vector
    v1={1.,2.,3.};
    cout<<"column vector v1:\n"<<v1<<endl;

    Matrix v2(1,3);    ////row vector
    v2={-3.,-2.,-1.};
    cout<<"row vector v2:\n"<<v2<<endl;


    Matrix v3=v1*v2;
    cout<<"v3=v1*v2, dimension: ["<<v3.m<<", "<<v3.n<<"]"<<endl;
    cout<<"v3 values:\n"<<v3<<endl;

    Matrix v4=v2*v1;
    cout<<"v4=v2*v1, dimension: ["<<v4.m<<", "<<v4.n<<"]"<<endl;
    cout<<"v4 values:\n"<<v4<<endl;
    
    
//    test identity, transpose, and block

    Matrix m6(3,3);
    cout<<"m6:\n"<<m6.Identity()<<endl;

    Matrix m7(4,2);
    m7 = {1,2,3,4,5,6,7,8};
    m7.Transpose();
    cout<<"m7.Transpose():\n"<<m7<<endl;

    cout<<"m2.Block(0,0,2,2):\n"<<m2.Block(0,0,2,2)<<std::endl;
    
    Matrix m8(2,2);
    m8 = {1,2,3,4};
    cout<<"m8.First_Single_Scalar(2, 3):\n"<<m8.Scalar_Identity(2,1,3)<<std::endl;

}
void Test_HW2() {
    ////test Gaussian Elimination
    Matrix A1(3, 3);
    Matrix b1(3, 1);
    Matrix x1(3, 1);
    A1 = { 1.,1.,1.,2.,2.,5.,4.,6.,8. };
    b1 = { 1.,2.,3. };
    x1 = Solve(A1, b1, x1);
    if ((A1*x1) == b1) cout << "Solve A1x=b1:\n " << x1 << endl;
    else cout << "Wrong Answer for A1x=b1" << endl;
    Matrix A2(3, 3);
    Matrix b2(3, 1);
    Matrix x2(3, 1);
    A2 = { 5.,1.,3.,4.,5.,3.,1.,5.,2. };
    b2 = { 3.,6.,-1. };
    x2 = Solve(A2, b2, x2);
    if (A2*x2 == b2) cout << "Solve A2x=b2:\n " << x2 << endl;
    else cout << "Wrong Answer for A2x=b2" << endl;
    Matrix A3(5, 5);
    Matrix b3(5, 1);
    Matrix x3(5, 1);
    A3 = { 2.,4.,5.,3.,2.,
        4.,8.,3.,4.,3.,
        3.,3.,2.,7.,2.,
        1.,2.,2.,1.,3.,
        3.,4.,2.,5.,7. };
    b3 = { 7.,-4.,-15.,14.,16. };
    x3 = Solve(A3, b3, x3);
    cout<<A3<<std::endl;
    cout<<b3<<std::endl;
    if (A3*x3 == b3) cout << "Solve A3x=b3:\n " << x3 << endl;
    else cout << "Wrong Answer for A3x=b3" << endl;
    Matrix A4(1000, 1000);
    Matrix b4(1000, 1);
    Matrix x4(1000, 1);
    for (int i = 1; i < A4.m -1; i++) {
        A4(i, i) = 2.;
        A4(i, i + 1) = -1.;
        A4(i, i - 1) = -1.;
    }
    A4(0, 0) = A4(A4.m-1, A4.n-1) = 2.;
    A4(0, 1) = A4(A4.m - 1, A4.n - 2) = -1.;
    for (int i = 0; i < b4.m; i++) { b4(i, 0) = (double)i/(double)(b4.m*b4.m); }
    x4 = Solve(A4, b4, x4);
    if (A4*x4 == b4) cout << "Solve A4x=b4:\n " << x4 << endl;
    else cout << "Wrong Answer for A4x=b4" << endl;
}
void Test_HW3()
{
    std::cout<<"Test sparse matrix"<<std::endl;
    SparseMatrix mtx(5,5);
    vector<tuple<int,int,double> > elements;
    elements.push_back(make_tuple<int,int,double>(0,0,7));
    elements.push_back(make_tuple<int,int,double>(0,1,5));
    elements.push_back(make_tuple<int,int,double>(1,0,1));
    elements.push_back(make_tuple<int,int,double>(1,2,3));
    elements.push_back(make_tuple<int,int,double>(2,3,5));
    elements.push_back(make_tuple<int,int,double>(2,4,4));
    elements.push_back(make_tuple<int,int,double>(3,3,1));
    elements.push_back(make_tuple<int,int,double>(4,1,7));
    elements.push_back(make_tuple<int,int,double>(4,4,3));
    
    mtx=elements;

    cout<<"sparse matrix:\n"<<mtx<<endl;

    Matrix v(5,1);v={1,2,3,4,5};
    Matrix prod(5,1);
    prod=mtx*v;
    cout<<"sparse matrix-vector multiplication:\n";
    cout<<prod<<endl;
//    cout<<"Jacobi iteration x is equal to"<<mtx.Jacobi_Solve(prod)<<std::endl;
}

void Test_HW4()
{
    std::cout<<"Test graph matrix"<<std::endl;
    Graph g;
    g.Add_Node(0.);
    g.Add_Node(1.);
    g.Add_Node(2.);
    g.Add_Node(3.);
    g.Add_Node(4.);
    g.Add_Node(5.);

    g.Add_Edge(0,1);
    g.Add_Edge(1,2);
    g.Add_Edge(1,3);
    g.Add_Edge(2,3);
    g.Add_Edge(2,4);
    g.Add_Edge(3,4);
    g.Add_Edge(4,5);
    
    Matrix adj_m;g.Adjacency_Matrix(adj_m);
    Matrix inc_m;g.Incidence_Matrix(inc_m);
    Matrix lap_m;g.Laplacian_Matrix(lap_m);

//    double energy=g.Dirichlet_Energy(g.node_values);

//    cout<<g<<endl;
    cout<<"Adjacency matrix\n"<<adj_m<<endl;
    cout<<"Incidency matrix\n"<<inc_m<<endl;
//    cout<<"Laplacian matrix\n"<<lap_m<<endl;
//    cout << "Dirichlet energy before smoothing: "<<energy<<endl;

//    g.Smooth_Node_Values(.1,10);
//    energy=g.Dirichlet_Energy(g.node_values);
//    cout<<"Dirichlet energy after smoothing: "<<energy<<endl;

}
void Test_ExtraCredit_HW4()
{
    std::cout<<"Extra Credit Test graph matrix"<<std::endl;
       SparseGraph g;
       g.Add_Node(0.);
       g.Add_Node(1.);
       g.Add_Node(2.);
       g.Add_Node(3.);
       g.Add_Node(4.);
       g.Add_Node(5.);

       g.Add_Edge(0,1);
       g.Add_Edge(1,2);
       g.Add_Edge(1,3);
       g.Add_Edge(2,3);
       g.Add_Edge(2,4);
       g.Add_Edge(3,4);
       g.Add_Edge(4,5);
    
    SparseMatrix adj_m;g.Adjacency_Matrix(adj_m);
    SparseMatrix inc_m;g.Incidence_Matrix(inc_m);
//    SparseMatrix lap_m;g.Laplacian_Matrix(lap_m);
//       double energy=g.Dirichlet_Energy(g.node_values);

       cout<<g<<endl;
       cout<<"Adjacency matrix\n"<<adj_m<<endl;
       cout<<"Incidency matrix\n"<<inc_m<<endl;
//       cout<<"Laplacian matrix\n"<<lap_m<<endl;
//       cout << "Dirichlet energy before smoothing: "<<energy<<endl;
//
//       g.Smooth_Node_Values(.1,10);
//       energy=g.Dirichlet_Energy(g.node_values);
//       cout<<"Dirichlet energy after smoothing: "<<energy<<endl;
}
void Test_Combo1()
{
    // Generate a random 30 by 10 Matrix A
    Matrix A1(30,10);
    A1.Random_Matrix_Generator();
    
    // Generate a random 30 -vector b.
    Matrix b1(30,1);
    b1.Random_Matrix_Generator();
    
    // Compute the least squares approximate solution xˆ = A†b
    Matrix x_hat = Solve_Least_Squares(A1, b1);
    cout<<"Least Square Solution Matrix x: \n"<<x_hat<<std::endl;
    
    cout<<"dif from actual and estimated: "<<b1-A1*x_hat<<std::endl;
    
    
    // Compute the associated residual norm squared ∥Axˆ − b∥2
    double r = Residual_Calculator(A1, b1, x_hat);
    cout<<"Residual Value: \n"<<r<<std::endl;
    
    // Generate three different random 10 -vectors d1, d2, d3,
    Matrix d1(10,1);
    Matrix d2(10,1);
    Matrix d3(10,1);
    d1.Random_Matrix_Generator();
    d2.Random_Matrix_Generator();
    d3.Random_Matrix_Generator();

    // Run Least Square Check
    // Verify that ∥A (xˆ + di) − b∥2 > ∥Axˆ − b∥2 holds
    if (Residual_Calculator(A1,b1,x_hat+d1)<r) cout<<"Least Square Calculation Failed! Be Better!"<<std::endl;
    if (Residual_Calculator(A1,b1,x_hat+d2)<r) cout<<"Least Square Calculation Failed! Be Better!"<<std::endl;
    if (Residual_Calculator(A1,b1,x_hat+d3)<r) cout<<"Least Square Calculation Failed! Be Better!"<<std::endl;
    else cout<<"Least Square Calculation successful!!!"<<std::endl;
    
    // Generate a random 20 × 10 matrix A and 20 -vector b
    Matrix A2(20,10);
    Matrix b2(20,1);
    A2.Random_Matrix_Generator();
    b2.Random_Matrix_Generator();
    int k =500; // number of iterations

    // Compute xˆ = A†b
    Matrix x = Richardson_Algorithm(A2, b2, k);
    cout<<"Least Square Iterative Solution Matrix x: \n"<<x<<std::endl;
}
void Test_Combo2()
{
    // Generate a random 20 × 10 matrix and a 20− vector b
    Matrix A(20,10);
    Matrix b(20,1);
    A.Random_Matrix_Generator();
    b.Random_Matrix_Generator();
    
    // Solve least squares problem using ATAx = ATb for comparison
    auto startLS = high_resolution_clock::now();
    Matrix x_comp = Solve_Least_Squares(A, b);
    auto stopLS = high_resolution_clock::now();
    auto durationLS = duration_cast<microseconds>(stopLS - startLS);
    cout<<"Solution using ATAx = ATb: "<<x_comp<<std::endl;
    cout <<"Least Squares Factorization: "<< durationLS.count() <<std::endl;
    
    // QR factorization to solve the least-squares solution as xˆ = R−1QTy
    auto startQR = high_resolution_clock::now();
    Matrix x = QR_Factorization(A, b);
    auto stopQR = high_resolution_clock::now();
    auto durationQR = duration_cast<microseconds>(stopQR - startQR);
    cout<<"Solution using QR Factorization:  "<<x<<std::endl;
    cout <<"QR Factorization: "<< durationQR.count() <<std::endl;
    
    // Show that Ax = QQTb
    Matrix Q = Gram_Smidth_Algorithm(A, b);
    Matrix QQTb = Q*Q.T()*b;
    
    // Compare two x values
    Matrix x1 = Solve_Least_Squares(A, b);
    Matrix x2 = Solve_Least_Squares(A, QQTb);
    
    if (x1 == x2)cout<<"Yes Ax = QQTb!"<<std::endl;
    else cout<<"No Ax != QQTb /n"<<std::endl;
    
        
    // Generate two random matrices for multiplication testing
    Matrix Z(1000,1000);
    Matrix Y(1000,1000);
    Z.Random_Matrix_Generator();
    Y.Random_Matrix_Generator();
    
    // Test run time for original Matrix Multiplication
    auto start1 = high_resolution_clock::now();
    Z.slow_multiplication(Y);
                
    auto stop1 = high_resolution_clock::now();
    auto duration1 = duration_cast<microseconds>(stop1 - start1);
    cout <<"Slow Matrix-Matrix Multiplication time: "<< duration1.count() <<" Microseconds"<< endl;

    
    // Test run time for optimized Matrix Multiplication
    auto start2 = high_resolution_clock::now();
    Z*Y;
    auto stop2 = high_resolution_clock::now();
    auto duration2 = duration_cast<microseconds>(stop2 - start2);
    cout <<"Improved Matrix-Matrix Multiplication time: "<< duration2.count() <<" Microseconds"<< endl;
}

void App_Test()
{
    // Function Approximator Task
    auto start_ = high_resolution_clock::now();
    testFunctionApproximator();
    auto stop_ = high_resolution_clock::now();
    auto duration_ = duration_cast<microseconds>(stop_ - start_);
    cout<<"Function Approximator Run time: "<<duration_.count()<<" Microseconds"<<endl;

    // Image Classifier Task
    auto start = high_resolution_clock::now();
    testImageClassifier();
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    cout<<"Image Classifier Run time: "<<duration.count()<<" Microseconds"<<endl;

}
double data_function(const double x) {return pow(x,3)+pow(x,2)+1.;}

void Project_Test()
{

    string data_dictionary = "/Users/robertdoherty/Desktop/MNIST_Sub";
    vector<pair<int, int>> regressor_feature_sizes={{1, 16}, {16, 16}, {16, 16}, {16, 1}};
    Regressor reg(regressor_feature_sizes,&data_function);
    reg.Train();
    vector<pair<int, int>> classifier_feature_sizes={{28*28+1, 256}, {256, 256}, {256, 256}, {256, 10}};
    Classifier cls(data_dictionary, classifier_feature_sizes);
    cls.Train();

}





int main()
{
    std::cout<<"Hello CS70!"<<std::endl;

//    Test_HW1();
//    Test_HW2();
//    Test_HW3();
//    Test_HW4();
//    Test_ExtraCredit_HW4();
//    Test_Combo1();
//    Test_Combo2();
//    App_Test();
    Project_Test();

    system("PAUSE");
};
