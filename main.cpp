#include <iostream>
#include "MyMatrix.h"

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
	
	//test subtractions
	Matrix m4=-m3;
	cout<<"m4=-m3, m4:\n"<<m4<<endl;
	
	m4=m1-m3;
	cout<<"m4=m1-m3,m4:\n"<<m4<<endl;

	m4-=m1;
	cout<<"m4-=m1, m4:\n"<<m4<<endl;

	//test matrix-scalar products
	double s=2;
	Matrix m5=m4*s;
	cout<<"m5=m4*s, m5:\n"<<m5<<endl;
	m5*=s;
	cout<<"m5*=s, m5:\n"<<m5<<endl;

	//check matrix-matrix multiplication
	Matrix v1(3,1);	//column vector
	v1={1.,2.,3.};
	cout<<"column vector v1:\n"<<v1<<endl;

	Matrix v2(1,3);	//row vector
	v2={-3.,-2.,-1.};
	cout<<"row vector v2:\n"<<v2<<endl;

	Matrix v3=v1*v2;
	cout<<"v3=v1*v2, dimension: ["<<v3.m<<", "<<v3.n<<"]"<<endl;
	cout<<"v3 values:\n"<<v3<<endl;

	Matrix v4=v2*v1;
	cout<<"v4=v2*v1, dimension: ["<<v4.m<<", "<<v4.n<<"]"<<endl;
	cout<<"v4 values:\n"<<v4<<endl;

	////test identity, transpose, and block
	Matrix m6(3,3);
	cout<<"m6:\n"<<m6.Identity()<<endl;

	Matrix m7(4,2);
	cout<<"m7.Transpose():\n"<<m7.Transpose()<<endl;

	cout<<"m2.Block(0,0,2,2):\n"<<m2.Block(0,0,2,2)<<std::endl;
    
    // test determinant function
    Matrix m8(2,2);
    m8(0,0) = 3;
    m8(0,1) = 5;
    m8(1,0) = 1;
    m8(1,1) = 4;
    cout<<m8<<endl; 
    cout<<"m8 determinant:\n"<<m8.Determinant2x2()<<endl; 
}

int main()
{
	std::cout<<"Hello CS70!"<<std::endl;

	Test_HW1();

	system("PAUSE");
}
