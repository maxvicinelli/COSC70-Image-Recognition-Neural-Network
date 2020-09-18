//////////////////////////////////////////////////////////////////////////
////Dartmouth CS70.01 final project
///Written by Robert Doherty and Max Vicinelli
////http://www.dartmouth.edu/~boolzhu/cosc70.01.html
////Linear algebra and vector math library
//////////////////////////////////////////////////////////////////////////

#ifndef __MyNeuralNetwork_h__
#define __MyNeuralNetwork_h__
#include "MyMatrix.h"
#include <utility>
#include <fstream>
#include "App_tasks.h"

using namespace std;

////Task 1: linear layer
class LinearLayer
{
private:
Matrix stored_input; //// Here we should store the input matrix A for Backward
public:
Matrix weight; //theta
Matrix weight_grad; //// record the gradient of the weight // gradient of theta



////linear layer constructor
explicit LinearLayer(const int _m=1,const int _n=1):
    weight(Matrix(_m,_n)),weight_grad(Matrix(_m,_n))
{
/* _m is the input hidden size and _n is the output hidden size
* "Kaiming initialization" is important for neural network to converge. The NN will not converge without it!
*/
    for (auto &item : this->weight.data) {
    item=(double)(rand()%20000-10000)/(double)10000*sqrt(6.0/(double)(this->weight.m)); //// Kaiming initialization
    }
}

    Matrix Forward(Matrix& input)
    {
    /* input.m is batch size and input.n is the #features.
    * 1) Store theinput in stored_data for Backward.
    * 2) Return input * weight.
    */
    //     cout<<input.n<<std::endl;
    assert(input.n==this->weight.m);
    //// Code start ////
    this->stored_input = input;
    Matrix output = input * this->weight;
    return output;
    //// Code end ////
    }

    ////BE CAREFUL! THIS IS THE MOST CONFUSING FUNCTION. YOU SHOULD READ THE OVERLEAF CAREFULLY BEFORE DIVING INTO THIS!
    Matrix Backward(Matrix& output_grad)
    {
    //// Code start ////
    assert(output_grad.n==this->weight.n);
    //Calculate the gradient of the output (the result of the Forward method) w.r.t. the **weight** and store the product of the gradient and output_grad in weight_grad
    this->weight_grad = this->stored_input.T() * output_grad;
    //Calculate the gradient of the output (the result of the Forward method) w.r.t. the **input** and return the product of the gradient and output_grad
    return output_grad * this->weight.T();
    //// Code end ////
    }
    };

////Task 2: non-linear activation
class ReLU
{
    private:
    Matrix stored_input; //// Here we should store the input matrix A for Backward
    public:

    ////ReLU layer constructor
    ReLU()=default;

    Matrix Forward(const Matrix& input)
    {
        /*
        * input_data.m is batch size and input.n is the #features.
        * This method returns the relu result for each element.
        */
        //// Code start ////
        //Store the input in this->stored_data for Backward.
        this->stored_input = input;
        
        //Go though each element in input and perform relu=max(0,x)
         Matrix output = input;
        for (int i=0; i<input.data.size(); i++){
        if (input.data[i] < 0) output.data[i] = 0;
        }
        // Return relu(input)
        return output;
        //// Code end ////
}

/// <#Description#>
/// @param output_grad <#output_grad description#>
    Matrix Backward(const Matrix& output_grad)
    {
        /*
        * TODO: returns the gradient of the input data
        */
        //Set the relu gradient equal to the input gradient
        Matrix grad_relu = output_grad;
        
        //Iterate through stored input data and compute the gradient either:
            // •grad(relu)=1 if relu(x)=x or grad(relu)=0 if relu(x)=0.
        for(int i=0;i<(this->stored_input.data.size());i++){
            //Multiply the relu derivative by the input gradient
            if (this->stored_input.data[i]>0)grad_relu.data[i]= grad_relu.data[i]*1;
            else grad_relu.data[i] = grad_relu.data[i]*0;
        }
        //Return the stored gradient by the input gradient
        return grad_relu;
        }
};
// Desert task: new non-linear activation layer
class FreLU
{
    private:
    Matrix stored_input; //// Here we should store the input matrix A for Backward
    public:

    ////ReLU layer constructor
    FreLU()=default;

    Matrix Forward(const Matrix& input)
    {
        /*
        * input_data.m is batch size and input.n is the #features.
        * This method returns the relu result for each element.
        */
       
        // Store the input in this->stored_data for Backward.
        this->stored_input = input;

        /* Go though each element in input and perform frelu(x):
        *            { x+b if x>0
        * frelu(x) = {
        *            { b if x<=0
        */
        Matrix output = input;
        for (int i=0; i<input.data.size(); i++){
        if (input.data[i] < 0) output.data[i] = 0;
        output.data[i] +=0.1; //Add bias term
        }

        return output;
        //// Code end ////
    }
    Matrix Backward(const Matrix& output_grad)
    {
    //Same thing as relu gradient because derivative x+b --> 1
    /* grad(relu)=1 if relu(x)=x
    * grad(relu)=0 if relu(x)=0
    * TODO: returns the gradient of the input data
    * ATTENTION: Do not forget to multiply the grad(relu) with the output_grad
    */
        Matrix grad_relu = output_grad;
        //Iterate through stored input data and compute the gradient either:
        // •grad(relu)=1 if relu(x)=x or grad(relu)=0 if relu(x)=0.
    for(int i=0;i<(this->stored_input.data.size());i++){
    if (this->stored_input.data[i]>0)grad_relu.data[i]= grad_relu.data[i]*1;
    else grad_relu.data[i] = grad_relu.data[i]*0;
    }

    return grad_relu;
    }
};

////Task 3: Loss function
class MSELoss
    {
    private:
    Matrix stored_data;
    public:

    ////cross entropy loss constructor
    MSELoss()= default;

    ////return the mse loss mean(y_j-y_pred_i)^2
    double Forward(Matrix& pred,const Matrix& truth)
    {
        // Calculate the mean square error
        double MSE =0.0;
        for (int i=0; i<truth.data.size();i++){
        MSE += pow(pred.data[i]-truth.data[i], 2);
        }
        //store the difference in this->stored_data for Backward.
        this->stored_data = pred-truth;
        //return MSE(X_pred, X_truth) = ||X_pred-X_truth||^2 / n
        return MSE/(stored_data.m);
        }

    ////return the gradient of the input data
    Matrix Backward()
    {
        //return the gradient of the MSE loss: grad(MSE) = 2(X_pred-X_truth) / n
        return (stored_data*(double)2)*((double)1/stored_data.m);
    }
};

////Task 4: Network architecture
class Network
{
    public:
    int n_layers=0;
    bool desert = true; //Boolean depending on if demonstrating desert functions
    vector<LinearLayer>linear_layers;
    vector<ReLU> activation_layers;
    vector<FreLU> desert_activation_layers;

    ////MNISTNetwork constructor
    Network(const vector<pair<int, int>>& feature_sizes) {
    assert(feature_sizes.size()!=0);
    for (int i=0;i<feature_sizes.size()-1;i++) {assert(feature_sizes[i].second==feature_sizes[i+1].first);}
    //Initialize the array for the linear layers with the feature size specified in the vector feature_sizes.
    for (int i=0; i<feature_sizes.size(); i++) {
    LinearLayer linearlayer = LinearLayer(feature_sizes[i].first, feature_sizes[i].second);
    linear_layers.push_back(linearlayer);
    }
    //Initialize the array for the non-linear FreLU layers, the number of which should be feature_size.size()-1.
    for (int j=0; j<feature_sizes.size()-1; j++) {
    FreLU frelu = FreLU();
    desert_activation_layers.push_back(frelu);
    }
    //Initialize the array for the non-linear ReLU layers, the number of which should be feature_size.size()-1.
    for (int j=0; j<feature_sizes.size()-1; j++) {
    ReLU relu = ReLU();
    activation_layers.push_back(relu);
        }

    //Total layers is equal to the total number of linear and activation layers
    n_layers = 2*(int)feature_sizes.size() - 1;
    }

    Matrix Forward(const Matrix& input) {
        // Propagate the input from the first layer to the last layer (before the loss function) by going through the forward functions of all the layers in linear_layers and activation_layers
        // ex. linear[0] -> activation[0] -> linear[1] ->activation[1] -> ... -> linear[k-2] -> activation[k-2] -> linear[k-1]
        Matrix input_ = input;
        int layers = (n_layers+1)/2;
        for (int i=0;i<layers; i++){
            //If you have reached the linear layer right before the loss function just forward the input through the linear layer
            if (i == layers-1) input_ = linear_layers[i].Forward(input_);
            else{ //otherwise pass input through the linear layer and then pass that output through the activation layer
                Matrix linear = linear_layers[i].Forward(input_);
                //Use ReLU or FreLU depending of if demonstrating desert task or not
                if (desert==true)input_ = desert_activation_layers[i].Forward(linear);
                else if (desert==false)input_ = activation_layers[i].Forward(linear);
        }
        }
        // Return output of input passed through all the layers of the network
        return input_;
    }

Matrix Backward(const Matrix& output_grad) {
    //Propagate the gradient from the last layer to the first layer by going through the backward functions of all the layers in linear_layers and activation_layers
    Matrix output = output_grad;
    int layers = (n_layers+1)/2;
    for (int i= layers-1; i>=0; i--){
        //If you have reached the starting linear layer stop.
        if (i == 0) output = linear_layers[i].Backward(output);
        else{
            Matrix linear_grad = linear_layers[i].Backward(output);
            //Use ReLU or FreLU depending of if demonstrating desert task or not
            if (desert==true)output = desert_activation_layers[i-1].Backward(linear_grad);
            else if (desert==false)output = activation_layers[i-1].Backward(linear_grad);
            }
    }
    //Return the gradient of the output
    return output;
}
};


//Desert Task: New Neural Network
class Desert_Network
{
    public:
    int n_layers=0;
    vector<LinearLayer>linear_layers;
    vector<ReLU> activation_layers;
    vector<FreLU> desert_activation_layers;

    ////MNISTNetwork constructor
    Desert_Network(const vector<pair<int, int>>& feature_sizes) {
    assert(feature_sizes.size()!=0);
    for (int i=0;i<feature_sizes.size()-1;i++) {assert(feature_sizes[i].second==feature_sizes[i+1].first);}
    //Initialize the array for the linear layers with the feature size specified in the vector feature_sizes.
    for (int i=0; i<feature_sizes.size(); i++) {
    LinearLayer linearlayer = LinearLayer(feature_sizes[i].first, feature_sizes[i].second);
    linear_layers.push_back(linearlayer);
    }
    //Initialize the array for the non-linear FreLU layers, the number of which should be feature_size.size()-1.
    for (int j=0; j<feature_sizes.size()-1; j++) {
        FreLU frelu = FreLU();
        desert_activation_layers.push_back(frelu);
    }
    //Initialize the array for the non-linear ReLU layers, the number of which should be feature_size.size()-1.
     for (int j=0; j<feature_sizes.size()-1; j++) {
        ReLU relu = ReLU();
        activation_layers.push_back(relu);
    }

    //Total layers is equal to the total number of linear and activation layers
    n_layers = 2*(int)feature_sizes.size() - 1;
    }

    Matrix Forward(const Matrix& input) {
        // Propagate the input from the first layer to the last layer (before the loss function) by going through the forward functions of all the layers in linear_layers and activation_layers
        // ex. linear[0] -> activation1[0]->activation2[0]->linear[1]->activation1[1]->activation2[1] ... -> linear[k-2] -> activation1[k-2] -> activation2[k-2] -> linear[k-1]
        Matrix input_ = input;
        int layers = (n_layers+1)/2;
        for (int i=0;i<layers; i++){
            //If you have reached the linear layer right before the loss function just forward the input through the linear layer
            if (i == layers-1) input_ = linear_layers[i].Forward(input_);
            else{
                //Pass input first to linear than frelu activation layer than relu activation layer
                Matrix linear = linear_layers[i].Forward(input_);
                input_ = desert_activation_layers[i].Forward(linear);
                input_ = activation_layers[i].Forward(input_);
            }
        }
        // Return output of input passed through all the layers of the network
        return input_;
        }

    Matrix Backward(const Matrix& output_grad) {
        //Propagate the gradient from the last layer to the first layer by going through the backward functions of all the layers in linear_layers and activation_layers
        Matrix output = output_grad;
        int layers = (n_layers+1)/2;
        for (int i= layers-1; i>=0; i--){
            //If you reach the first linear layer stop.
            if (i == 0) output = linear_layers[i].Backward(output);
            else{
                Matrix linear_grad = linear_layers[i].Backward(output);
                output = desert_activation_layers[i-1].Backward(linear_grad);
                output = activation_layers[i-1].Backward(output);
                }
        }
        //Return the gradient of the output
        return output;
    }
    };


////Task 5: Matrix slicing
Matrix Matrix_Slice(const Matrix& A, const int start, const int end)
{
    //We need to slice the matrix for batch stochastic gradient decent
    Matrix a = A;
    Matrix slice = a.Slice(start, end-1);
    //Return a matrix with rows of the input A from row 'start' to row 'end-1'.
    return slice;
}

////Task 6: Regression
class Regressor
{
    public:

    //Network net;
    Network net;
    MSELoss loss_function=MSELoss();
    Matrix train_data;
    Matrix train_targets;
    Matrix test_data;
    Matrix test_targets;
    double learning_rate=1e-3;
    int max_epoch=200;
    int batch_size=32;
    //
    //
    ////Classifier constructor
    Regressor(vector<pair<int, int>> feature_sizes, double (*unknown_function)(const double)):
        //     net(Desert_Network(feature_sizes)),
        net(Network(feature_sizes)),
        train_data(Matrix(1000,1)), train_targets(Matrix(1000,1)),
        test_data(Matrix(200,1)), test_targets(Matrix(200,1))
        {
        for (int i=0;i<1000;i++) {
        double x=(double)(rand()%20000-10000)/(double)10000;
        double y=unknown_function(x);
        train_data(i, 0)=x;
        train_targets(i, 0)=y;
        }

        for (int i=0;i<200;i++) {
            double x=(double)(rand()%20000-10000)/(double)10000;
            double y=unknown_function(x);
            test_data(i, 0)=x;
            test_targets(i, 0)=y;
        }
        }

    //// Here we train the network using gradient descent
    double Train_One_Epoch()
    {
        double loss=0;
        int n_loop=this->train_data.m/this->batch_size;
        for (int i=1;i<n_loop;i++){
            Matrix batch_data=Matrix_Slice(this->train_data, (i-1)*this->batch_size, i*this->batch_size);
            Matrix batch_targets=Matrix_Slice(this->train_targets, (i-1)*this->batch_size, i*this->batch_size);

             //Forward the data to the network.
            Matrix pred = this->net.Forward(batch_data);
            //Update Loss
             loss += this->loss_function.Forward(pred, batch_targets);
            //Forward the result to the loss function
             Matrix pred_grad = this->loss_function.Backward();
            //Backward.
             net.Backward(pred_grad);   //// we do not need the gradient for train_data,but just the parameters.
            //Update weights
             for (auto& item : this->net.linear_layers) {
                 item.weight -= item.weight_grad*(this->learning_rate);
             }
        }
        return loss/(double)n_loop;
    }

    double Test()
    {
        Matrix pred=this->net.Forward(this->test_data);
        double loss=this->loss_function.Forward(pred,this->test_targets);
        return loss;
    }

    void Train()
    {
        for (int i=0;i<this->max_epoch;i++) {
        double train_loss=Train_One_Epoch();
        double test_loss=Test();
        std::cout<<"Epoch: "<<(i+1)<<"/"<<this->max_epoch<<" | Train loss: "<<train_loss<<" | Test loss: "<<test_loss<<std::endl;
        }
    }
};


Matrix One_Hot_Encode(vector<int> labels, int classes=10)
{
    /* Make the labels one-hot.
    * For example, if there are 5 classes {0, 1, 2, 3, 4} then
    * [0, 2, 4] -> [[1, 0, 0, 0, 0],
    * [0, 0, 1, 0, 0],
    * [0, 0, 0, 0, 1]]
    */
    //Build matrix with size equal to the number of images by the number of classes
    Matrix one_hot((int)labels.size(),classes);
    //Iterate through labels size and set the collumn corresponding to the label equal to 1
    for (int i=0;i<(int)labels.size();i++){
        int j = labels[i];
        one_hot(i,j) = 1;
    }
    return one_hot;
}


class Classifier
{
public:

    // Network net;
    Desert_Network net;
    MSELoss loss_function=MSELoss();
    Matrix train_data; //// The shape should be (m=n_samples,n=28^2)
    vector<int> train_labels;
    Matrix test_data;
    vector<int> test_labels;
    double learning_rate=1e-3;
    int max_epoch=200;
    int batch_size=32;

////Classifier constructor
Classifier(const string& data_dict,const vector<pair<int, int>>& feature_sizes):
    // net(Desert_Network(feature_sizes)),
    net(Desert_Network(feature_sizes)),
    train_data(Matrix(1000,28*28+1)),test_data(Matrix(200,28*28+1))
    {
        Matrix noise(1000,28*28+1);
        noise.Noise_Matrix_Generator(100);
        int tempint;
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
        file.close();

        file = ifstream(data_dict+"/train_labels.txt");
        if (!file.is_open()) std::cout<<"file does not exist"<<std::endl;
        for(int i=0;i<1000;i++) if(file >> tempint) train_labels.push_back(tempint);
        file.close();

        file = ifstream(data_dict+"/test_data.txt");
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

        file = ifstream(data_dict+"/test_labels.txt");
        if (!file.is_open()) std::cout<<"file does not exist"<<std::endl;
        for(int i=0;i<200;i++) if(file >> tempint) test_labels.push_back(tempint);
        file.close();
    }

    double Train_One_Epoch()
    {
    double loss=0;
    int n_loop=this->train_data.m/this->batch_size;
    for (int i=1;i<n_loop;i++){
        Matrix batch_data=Matrix_Slice(this->train_data, (i-1)*this->batch_size, i*this->batch_size);
        auto start=this->train_labels.begin()+(i-1)*this->batch_size;
        vector<int> batch_labels = vector<int>(start, start+this->batch_size);

        //Make labels matrix
        Matrix batch_targets = One_Hot_Encode(batch_labels);
        //Forward the data to the network.
        Matrix pred = this->net.Forward(batch_data);
        //Update Loss
         loss += this->loss_function.Forward(pred, batch_targets);
        //Forward the result to the loss function
         Matrix pred_grad = this->loss_function.Backward();
        //Backward.
         net.Backward(pred_grad);   //// we do not need the gradient for train_data,but just the parameters.
        //Update weights
         for (auto& item : this->net.linear_layers) {
             item.weight -= item.weight_grad*(this->learning_rate);
         }

    }
    return loss/(double)n_loop;
    }

double Test()
{
    Matrix score=this->net.Forward(this->test_data); //// the class with max score is our predicted label
    double accuracy=0;
    for (int i=0;i<score.m;i++){
        int max_index=0;
        for (int j=0;j<score.n;j++){ if (score(i,j)>score(i,max_index)) {max_index=j;} }
        if (max_index==test_labels[i]) {accuracy+=1;}
    }
    return accuracy/(double)score.m;
}

void Train()
{
    for (int i=0;i<this->max_epoch;i++) {
        double loss=Train_One_Epoch();
        double accuracy=Test();
        std::cout<<"Epoch: "<<(i+1)<<"/"<<this->max_epoch<<" | Train loss: "<<loss<<" | Test Accuracy: "<<accuracy<<std::endl;
    }
    }
};

#endif
