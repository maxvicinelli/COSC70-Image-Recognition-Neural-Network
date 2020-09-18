//////////////////////////////////////////////////////////////////////////
////Dartmouth CS70.01 final project
////http://www.dartmouth.edu/~boolzhu/cosc70.01.html
////Linear algebra and vector math library
//////////////////////////////////////////////////////////////////////////

#ifndef App_tasks_h
#define App_tasks_h
#include "MyMatrix.h"
#include <utility>
#include <fstream>

using namespace std;

Matrix train_data;  //// The shape should be (m=n_samples,n=28^2)
vector<int> train_labels;
Matrix test_data;
vector<int> test_labels;

//Appetizer Tasks
// Use Least Squares to approximate f(x) implemented in function pointer "unknown function"
// Need to decide  f1(x), f2(x), f3(x), ..., fn(x)

// Actual F(x) = 2x^3 + 3x^2
double F_x(const double x) {
    double y = 3*pow(x, 2) + 2*pow(x, 3);
    return y;
}

// returns matrix of predicted values of y
// a is all the x values of the points
Matrix Function_Approximation(vector<double>a)
  {
      Matrix A((int)a.size(), 4);
      Matrix Y((int)a.size(),1);
      
      //Approximate f(x) --> f0(x)=1, f1(x)=x, f2(x)=x^2, f3(x) = x^3
      for (int i=0; i<a.size();i++){
          A(i,0) = 1;
          A(i,1) = a[i];
          A(i,2) = pow(a[i],2);
          A(i,3) = pow(a[i],3);
          Y(i,0) = F_x(a[i]);
      }
      

      // Use A^TAx = A^Tb to solver for theta
      return Solve_Least_Squares(A, Y);
  }

// Calculate the mean square error for theta derived above
double MSE(Matrix& theta, vector<double>a, int n) {
    Matrix A((int)a.size(), 4);
    Matrix y((int)a.size(),1);

    // Create matrix A with values f0(x)=1, f1(x)=x, f2(x)=x^2, f3(x) = x^3 for each collumn
    // Create matrix y with actual values for x
    for (int i=0; i<a.size();i++){
       A(i,0) = 1;
       A(i,1) = a[i];
       A(i,2) = pow(a[i],2);
       A(i,3) = pow(a[i],3);
       y(i,0) = F_x(a[i]);
    }

    // Approximate y using theta values calculated in function approximation
    Matrix y_hat = A*theta;

    // Calculate the mean square error
    double MSE =0.0;
    for (int i=0; i<y.m;i++){
       MSE += pow(y(i,0)-y_hat(i,0), 2);
    }
    return MSE/n;
}

Matrix Build_Image_Classifier(string data_dict, Matrix& noise){
    Matrix train_data(1000, 28*28+1);  //// The shape should be (m=n_samples,n=28^2)    int tempint;
    Matrix train_labels = Matrix(1000, 1);
    int tempint;
    
    // Creating train data matrix 1000 by 785
    ifstream file(data_dict+"/train_data.txt");
    if (!file.is_open()) std::cout<<"file does not exist"<<std::endl;
    for(int i=0;i<1000;i++){
        for(int j=0; j< 28*28;j++){
            if(file >> tempint){
                train_data(i,j)=(double)tempint/255.0; // redefining temp int (data point in file)
                if (j%783 == 0)train_data(i,j+1) = (double)1 + noise(i,j+1);
            }
            train_data(i,j) += noise(i,j); // adding noise
        }
    }



    // Read through train labels data file and create 1000x1 matrix of correct labels for each image
    file = ifstream(data_dict+"/train_labels.txt");
    if (!file.is_open()) std::cout<<"file does not exist"<<std::endl;
    for(int i=0;i<1000;i++) if(file >> tempint) train_labels.data[i]=(tempint);
    file.close();

    // Create new 1000x10 matrix of train labels with 1's in column corresponding to number of image and -1's otherwise
    Matrix vector_train_labels(1000,10);
    for (int i=0; i<1000;i++){
       for (int j=0; j<10;j++ ){
           vector_train_labels(i,j)=-1;
       }
       int j = train_labels.data[i];
       vector_train_labels(i, j) = 1;
    }
       

    // Find Q and R matrices using Gram-Schmidt
    Matrix Q(train_data.m,train_data.n);
    Matrix R(train_data.n, train_data.n);
    train_data.Gram_Schmidt(Q, R);
    
    // do QR Factorization to find theta for each train labels vector
    vector<Matrix> theta_list;
    for (int j=0; j<vector_train_labels.n;j++){
       Matrix y(1000,1);
       for (int i=0; i<vector_train_labels.m; i++){
           y(i,0) = vector_train_labels(i,j);
       }
       Matrix theta_i = train_data.QRFactorization(Q, R, y);
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

// returns number of correct predictions
double compare(Matrix prediction, Matrix actual) {
   double num_correct = 0;
   for (int i=0; i<prediction.m; i++) {
       if (prediction(i, 0) == actual(i, 0)) {
           num_correct++;
       }
   }
   return num_correct;
}

// Randomly create x values, test function approximator
void testFunctionApproximator(){
    // Function Approximator Task
    vector<double> A;
    for (int i=0; i<200; i++)A.push_back(rand()%100);
    Matrix theta = Function_Approximation(A);
    cout<<theta<<std::endl;
       
    // test theta with different inputs
    vector<double> A_test;
    for (int i=0; i<200; i++)A_test.push_back(rand()%100);
    cout<<"MSE Value: "<< MSE(theta, A, 200)<<endl;
}

// builds test data and train data matrices, tests predicted values
// returns accuracy percentage
double helperTestImageClassifier(Matrix theta, const string& data_dict, Matrix& noise) {
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
           test_data(i,j)=(double)tempint/255.0;
               if (j%783 == 0)test_data(i,j+1) = (double)1 + noise(i,j+1);
           }
           test_data(i,j) += noise(i,j);
       }
   }
           
   file.close();
   

   // build test labels matrix
   file = ifstream(data_dict+"/test_labels.txt");
   if (!file.is_open()) std::cout<<"file does not exist"<<std::endl;
   for(int i=0;i<200;i++) if(file >> tempint) test_labels.data[i]=(tempint);
   file.close();
   
   // use theta to find prediction vector for each image
   prediction_vectors = test_data*theta;
   
   // find max in each prediction vector and add index to prediction labels
   for (int i=0; i<prediction_vectors.m; i++) {
       int max_column = 0;
       for (int j=1; j<prediction_vectors.n; j++) {
           if (prediction_vectors(i, max_column) < prediction_vectors(i,j)) max_column = j;
           
       }
       prediction_labels(i, 0) = max_column;
   }
   return compare(prediction_labels, test_labels);
}

void testImageClassifier() {
   string data_dictionary = "/Users/Robertdoherty/Desktop/MNIST_Sub";
   Matrix noise(1000, 28*28+1);
   noise.Noise_Matrix_Generator(10);
   Matrix theta = Build_Image_Classifier(data_dictionary, noise);
   double num_correct = helperTestImageClassifier(theta, data_dictionary, noise);
   cout<<"Correctly Predicted: "<< num_correct<<std::endl;
   cout<<"Image classifier success rate: \n"<<num_correct/(double)200<<endl;
}



#endif /* App_tasks_h */
