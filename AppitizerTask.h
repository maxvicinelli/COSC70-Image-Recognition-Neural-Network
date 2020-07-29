//
//  AppitizerTask.h
//  Programming Assignments
//
//  Created by Max Vicinelli on 3/2/20.
//  Copyright © 2020 Max Vicinelli. All rights reserved.
//

#ifndef AppitizerTask_h
#define AppitizerTask_h

#include "MyMatrix.h"
#include <utility>
#include <fstream>

// Our function f(x) = 3x^2 + 2x^3

double f_x(double x) {
    return 3*x*x + 2*x*x*x;
}

// returns theta matrix
// n is number of data points
// k is number of basis functions
Matrix buildTheta(int n, int k) {
    Matrix data = Matrix(n, 2); 
    for (int h=0; h<n; h++) {
        data(h, 0) = h;
        data(h, 1) = f_x(h);
    }
    
    Matrix A = Matrix(n,k);
    for (int i=0; i<n; i++) {
        for (int j=0; j<k; j++) {
            double x = data(i,0);
            A(i,j) = pow(x, j);
        }
    }
    
    Matrix y = Matrix(n, 1);
    for (int i=0; i<n; i++) {
        y(i,0) = data(i, 1);
    }
    
    Matrix theta = leastSquaresSolver(A, y);
    return theta;
}

// k is number of basis functions
// h is number of points we want to test
// returns mean squared error
double testTheta(Matrix theta, int k, int h) {
    
    // first column stores x's from 0 to h
    // second column is predicted value of y from f hat
    Matrix testData = Matrix(h, 2);
    for (int i=0; i<h; i++) {
        testData(i,0) = i;
    }
    
    // calculate f hat for each x
    for (int i=0; i<h; i++) {
        double fHat = 0;
        for (int j=0; j<k; j++) {
            fHat += theta(j, 0)*pow(testData(i,0), j);
        }
        testData(i, 1) = fHat;
    }
    
    // calculate mse
    double mse = 0;
    for (int i=0; i<h; i++) {
        mse += pow(abs(testData(i, 1) - f_x(i)), 2)/h;
    }
    
    return mse;
}


void testFunctionApproximator() {
    int n = 100;
    int k = 4;
    Matrix theta = buildTheta(n, k);
    cout<<theta<<endl;
    cout<<"MSE is:"<<endl;
    cout<<testTheta(theta, theta.m, 200)<<endl;

}

// Second appitizer task:
// Build theta matrix

Matrix buildImageClassifier(const string& data_dict) {
    Matrix train_data = Matrix(1000, 785); 
    
    // Creating train data matrix 1000 by 785
    int tempint;
    ifstream file(data_dict+"/train_data.txt");
    if (!file.is_open()) std::cout<<"file does not exist"<<std::endl;
    for(int i=0;i<1000;i++){
        for(int j=0; j<28*28;j++){
            if(file >> tempint){
                train_data(i,j)=(double)tempint/255.0; // redefining temp int (data point in file)
                if (j%783 == 0)train_data(i,j+1) =1;
                
            }
            train_data(i,j) += rand()%(1/1000);
        }
    }
    
    Matrix train_labels = Matrix(1000, 10);
    
    file = ifstream(data_dict+"/train_labels.txt");
    if (!file.is_open()) std::cout<<"file does not exist"<<std::endl;
    for(int i=0;i<1000;i++) if(file >> tempint)
        train_labels(i, tempint) = 1;
    file.close();
    
     
    
    
    // Break up 1000x10 train labels matrix into 10 1000x1 matrices
    vector<Matrix> Y;
    for (int j=0; j<train_labels.n;j++){
        Matrix y(1000,1);
        for (int i=0; i<train_labels.m; i++){
            y(i,0) = train_labels(i,j);
        }
        Y.push_back(y);
    }
    
    cout<<"beginning least squares solver"<<endl;
    
    // do least squares to find theta for each train labels vector
    vector<Matrix> theta_list;
    for (int i=0; i<Y.size();i++){
        Matrix theta_i = leastSquaresSolver(train_data, Y[i]);
        theta_list.push_back(theta_i);
    }

    //Combine all 10 theta vectors into 785x10
    Matrix theta(785,10);
    for (int j=0; j<theta_list.size();j++){
        for (int i=0; i<785;i++){
            theta(i,j) = theta_list[j](i,0);
        }
    }
    
    return theta;
}

double compare(Matrix prediction, Matrix actual) {
    double num_correct = 0;
    for (int i=0; i<prediction.m; i++) {
        cout<<"prediction: "<<prediction(i,0)<<endl;
        cout<<"actual: "<<actual(i,0)<<endl;
        if (prediction(i, 0) == actual(i, 0)) {
            cout<<"added one"<<endl; 
            num_correct++;
        }
    }
    return num_correct;
}

double testImageClassifier(Matrix theta, const string& data_dict) {
    Matrix test_data = Matrix(200, 785); // holds pixels for each image
    Matrix test_labels = Matrix(200, 1); // holds true label for each image
    Matrix prediction_vectors = Matrix(200, 10); // matrix containing predicted labels, 0-9 for each image
    Matrix prediction_labels = Matrix(200, 1); // holds predicted label for each image
    
    //  build test data matrix
    int tempint;
    ifstream file(data_dict+"/test_data.txt");
    if (!file.is_open()) std::cout<<"file does not exist"<<std::endl;
    for(int i=0;i<200;i++){
        for(int j=0; j< 28*28;j++){
            if(file >> tempint){
                test_data(i,j)=(double)tempint/255.0; // redefining temp int (data point in file)
                if (j%783 == 0)test_data(i,j+1)=1;
            }
        }
    }
    cout<<test_data<<endl;
    cout<<"-----"<<endl;
    cout<<theta<<endl;

    // build test labels matrix
    file = ifstream(data_dict+"/test_labels.txt");
    if (!file.is_open()) std::cout<<"file does not exist"<<std::endl;
    for(int i=0;i<200;i++) if(file >> tempint) test_labels.data[i]=(tempint);
    file.close();
    
    // use theta to find prediction vector for each image
    prediction_vectors = test_data*theta;
    cout<<prediction_vectors<<endl;
    
    // find max in each prediction vector and add index to prediction labels
    for (int i=0; i<prediction_vectors.m; i++) {
        int max_column = 0;
        for (int j=1; j<prediction_vectors.n; j++) {
            if (prediction_vectors(i, max_column) < prediction_vectors(i,j)) {
                max_column = j;
            }
        }
        prediction_labels(i, 0) = max_column;
    }
    return compare(prediction_labels, test_labels);
}

void mainImageTest() {
    string data_dictionary = "/Users/maxvicinelli/Desktop/MNIST_Sub/";
    Matrix theta = buildImageClassifier(data_dictionary);
    double num_correct = testImageClassifier(theta, data_dictionary);
    
    cout<<"Correctly predicted: "<<num_correct<<" images"<<endl;

    cout<<"Image classifier success rate: \n"<<num_correct/200<<endl;
}




#endif /* AppitizerTask_h */
