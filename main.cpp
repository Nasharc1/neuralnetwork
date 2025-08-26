#include <iostream>
#include <vector>
#include "project2_a.h"
#include <fstream>
#include <stdexcept>

// Test feed_forward function
void test_feed_forward() {
    // Create a simple network: 2 inputs, 1 hidden layer with 2 neurons, 1 output
    TanhActivationFunction* activation_function = new TanhActivationFunction();
    unsigned n_input = 2;
    std::vector<std::pair<unsigned, ActivationFunction*>> non_input_layer = {
        {2, activation_function}, // Hidden layer
        {1, activation_function}  // Output layer
    };

    NeuralNetwork network(n_input, non_input_layer);

    // Manually set weights and biases for reproducibility
    network.initialise_parameters(0.0, 0.1);
    
    // Input
    DoubleVector input(2);
    input[0] = 1.0; input[1] = 2.0;

    // Run feed forward
    DoubleVector output;
    network.feed_forward(input, output);

    std::cout << "Output: " << output[0] << std::endl;

    delete activation_function;
}

// Test cost function
void test_cost() {
    TanhActivationFunction* activation_function = new TanhActivationFunction();
    unsigned n_input = 2;
    std::vector<std::pair<unsigned, ActivationFunction*>> non_input_layer = {
        {1, activation_function}
    };

    NeuralNetwork network(n_input, non_input_layer);

    // Input and target output
    DoubleVector input(2);
    input[0] = 1.0; input[1] = 2.0;
    DoubleVector target_output(1);
    target_output[0] = 0.5;

    // Compute cost
    double cost_value = network.cost(input, target_output);
    std::cout << "Cost: " << cost_value << std::endl;

    delete activation_function;
}

// Test cost_for_training_data function
void test_cost_for_training_data() {
    // Create training data
    std::vector<std::pair<DoubleVector, DoubleVector>> training_data;

    DoubleVector input1(2);
    input1[0] = 1.0; input1[1] = 2.0;
    DoubleVector target1(1);
    target1[0] = 0.5;
    training_data.push_back({input1, target1});

    DoubleVector input2(2);
    input2[0] = 2.0; input2[1] = 3.0;
    DoubleVector target2(1);
    target2[0] = 0.3;
    training_data.push_back({input2, target2});

    DoubleVector input3(2);
    input3[0] = 1.5; input3[1] = 2.5;
    DoubleVector target3(1);
    target3[0] = 0.4;
    training_data.push_back({input3, target3});

    // Create a simple network
    TanhActivationFunction* activation_function = new TanhActivationFunction();
    unsigned n_input = 2;
    std::vector<std::pair<unsigned, ActivationFunction*>> non_input_layer = {
        {1, activation_function}
    };

    NeuralNetwork network(n_input, non_input_layer);

    // Compute cost for training data
    double average_cost = network.cost_for_training_data(training_data);
    std::cout << "Average Cost for Training Data: " << average_cost << std::endl;

    delete activation_function;
}

// Test edge cases with feed_forward
void test_feed_forward_edge_cases() {
    TanhActivationFunction* activation_function = new TanhActivationFunction();
    unsigned n_input = 2;
    std::vector<std::pair<unsigned, ActivationFunction*>> non_input_layer = {
        {2, activation_function},
        {1, activation_function}
    };

    NeuralNetwork network(n_input, non_input_layer);

    // Zero inputs
    DoubleVector zero_input(2);
    zero_input[0] = 0.0; zero_input[1] = 0.0;
    DoubleVector zero_output;
    network.feed_forward(zero_input, zero_output);
    std::cout << "Output for zero inputs: " << zero_output[0] << std::endl;

    // Large inputs
    DoubleVector large_input(2);
    large_input[0] = 1e6; large_input[1] = 1e6;
    DoubleVector large_output;
    network.feed_forward(large_input, large_output);
    std::cout << "Output for large inputs: " << large_output[0] << std::endl;

    // Small inputs
    DoubleVector small_input(2);
    small_input[0] = 1e-6; small_input[1] = 1e-6;
    DoubleVector small_output;
    network.feed_forward(small_input, small_output);
    std::cout << "Output for small inputs: " << small_output[0] << std::endl;

    delete activation_function;
}

void test_output_training_data() {
    std::vector<std::pair<DoubleVector, DoubleVector>> training_data;
    
    DoubleVector input1(2);
    input1[0] = 1.0; input1[1] = 2.0;
    DoubleVector target1(1);
    target1[0] = 0.5;
    training_data.push_back({input1, target1});

    DoubleVector input2(2);
    input2[0] = 2.0; input2[1] = 3.0;
    DoubleVector target2(1);
    target2[0] = 0.3;
    training_data.push_back({input2, target2});

    DoubleVector input3(2);
    input3[0] = 1.5; input3[1] = 2.5;
    DoubleVector target3(1);
    target3[0] = 0.4;
    training_data.push_back({input3, target3});
    
    NeuralNetwork::output_training_data("output_training_data.txt", training_data);
    std::cout << "Output training data to file." << std::endl;
}

void test_read_training_data() {
    std::vector<std::pair<DoubleVector, DoubleVector>> training_data;
    NeuralNetwork::read_training_data("project_training_data.dat", training_data);
    std::cout << "Read training data successfully. Entries: " << training_data.size() << std::endl;
}

// Test train function
void test_train() {
    // Read actual training data
    std::vector<std::pair<DoubleVector, DoubleVector>> training_data;
    NeuralNetwork::read_training_data("project_training_data.dat", training_data);

    // Create a network for the task
    TanhActivationFunction* activation_function = new TanhActivationFunction();
    unsigned n_input = 2;
    std::vector<std::pair<unsigned, ActivationFunction*>> non_input_layer = {
        {3, activation_function}, // First hidden layer
        {3, activation_function}, // Second hidden layer
        {1, activation_function}  // Output layer
    };

    NeuralNetwork network(n_input, non_input_layer);

    // Initialize parameters
    network.initialise_parameters(0.0, 0.1);

    // Train network and log convergence history
    network.train(training_data, 0.1, 1e-4, 10000, "convergence_history.txt");
    
    // Print final cost
    double final_cost = network.cost_for_training_data(training_data);
    std::cout << "Final Cost After Training: " << final_cost << std::endl;

    // Save trained network parameters
    network.write_parameters_to_disk("trained_network_params.txt");

    // Generate decision boundary data
    network.generate_decision_boundary_data();

    delete activation_function;
}

// Test initialise_parameters function
void test_initialise_parameters() {
    TanhActivationFunction* activation_function = new TanhActivationFunction();
    unsigned n_input = 2;
    std::vector<std::pair<unsigned, ActivationFunction*>> non_input_layer = {
        {2, activation_function},
        {1, activation_function}
    };

    NeuralNetwork network(n_input, non_input_layer);
    network.initialise_parameters(0.0, 0.1);

    std::cout << "Initialization complete. Parameters are randomly set." << std::endl;

    delete activation_function;
}

int main() {
    try {
        // Run all tests
        test_feed_forward();
        test_cost();
        test_cost_for_training_data();
        test_feed_forward_edge_cases();
        test_read_training_data();
        test_output_training_data();
        
        // Main training and visualization test
        test_train();
        
        test_initialise_parameters();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    

    return 0;
}