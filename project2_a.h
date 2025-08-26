#ifndef PROJECT2_A_H
#define PROJECT2_A_H

#include "dense_linear_algebra.h"
#include <memory>
#include <vector>
#include <iostream>
#include <fstream>
#include <random>
#include <limits>
#include <string>
#include <stdexcept>


using BasicDenseLinearAlgebra::DoubleVector;
using BasicDenseLinearAlgebra::DoubleMatrix;

// RandomNumber namespace
namespace RandomNumber {
    static std::mt19937 Random_number_generator{std::random_device{}()};
}

class ActivationFunction {
public:
    virtual ~ActivationFunction() {}
    virtual std::string name() const = 0;
    virtual double sigma(const double& x) = 0;
    virtual double dsigma(const double& x) {
        double fd_step = 1.0e-8;
        double sigma_ref = sigma(x);
        double sigma_pls = sigma(x + fd_step);
        return (sigma_pls - sigma_ref) / fd_step;
    }
};

class TanhActivationFunction : public ActivationFunction {
public:
    std::string name() const override { return "TanhActivationFunction"; }
    double sigma(const double& x) override { return std::tanh(x); }
    double dsigma(const double& x) override { return 1.0 / std::cosh(x) / std::cosh(x); }
};

class NeuralNetworkLayer {
public:
    NeuralNetworkLayer(unsigned input_size, unsigned output_size, ActivationFunction* act)
        : weights(output_size, input_size), biases(output_size), activation(act) {}

    void feed_forward(const DoubleVector& input, DoubleVector& z, DoubleVector& output) const {
        z.resize(weights.n());
        for (unsigned i = 0; i < weights.n(); ++i) {
            z[i] = biases[i];
            for (unsigned j = 0; j < weights.m(); ++j) {
                z[i] += weights(i, j) * input[j];
            }
        }
        output.resize(z.n());
        for (unsigned i = 0; i < z.n(); ++i) {
            output[i] = activation->sigma(z[i]);
        }
    }

    void initialise_parameters(const double& mean, const double& std_dev) {
        std::normal_distribution<> dist(mean, std_dev);
        for (unsigned i = 0; i < weights.n(); ++i) {
            for (unsigned j = 0; j < weights.m(); ++j) {
                weights(i, j) = dist(RandomNumber::Random_number_generator);
            }
            biases[i] = dist(RandomNumber::Random_number_generator);
        }
    }

    void write_parameters(std::ofstream& file) const {
        file << activation->name() << "\n";
        file << weights.m() << "\n" << weights.n() << "\n";
        for (unsigned i = 0; i < biases.n(); ++i) {
            file << i << " " << biases[i] << "\n";
        }
        for (unsigned i = 0; i < weights.n(); ++i) {
            for (unsigned j = 0; j < weights.m(); ++j) {
                file << i << " " << j << " " << weights(i, j) << "\n";
            }
        }
    }

    const DoubleMatrix& get_weights() const { return weights; }
    const DoubleVector& get_biases() const { return biases; }
    ActivationFunction* get_activation() const { return activation; }
// Add these methods inside the NeuralNetwork class, before the private section
void read_training_data(const std::string& filename, 
                        std::vector<std::pair<DoubleVector, DoubleVector>>& training_data) {
    std::ifstream file(filename);
    if (!file) {
        throw std::runtime_error("Cannot open training data file: " + filename);
    }

    training_data.clear();
    double input1, input2, output;
    while (file >> input1 >> input2 >> output) {
        DoubleVector input(2);
        input[0] = input1;
        input[1] = input2;
        
        DoubleVector target(1);
        target[0] = output;
        
        training_data.push_back({input, target});
    }

    if (training_data.empty()) {
        throw std::runtime_error("No training data found in file");
    }
}

void output_training_data(const std::string& filename, 
                          const std::vector<std::pair<DoubleVector, DoubleVector>>& training_data) {
    std::ofstream file(filename);
    if (!file) {
        throw std::runtime_error("Cannot open output file: " + filename);
    }

    for (const auto& data_point : training_data) {
        const DoubleVector& input = data_point.first;
        const DoubleVector& output = data_point.second;
        
        file << input[0] << " " << input[1] << " " << output[0] << "\n";
    }
}


private:
    DoubleMatrix weights;
    DoubleVector biases;
    ActivationFunction* activation;
};

class NeuralNetwork {
public:
    NeuralNetwork(const unsigned& n_input,
                  const std::vector<std::pair<unsigned, ActivationFunction*>>& non_input_layer) {
        input_size = n_input;
        unsigned prev_size = n_input;
        for (const auto& layer_info : non_input_layer) {
            unsigned size = layer_info.first;
            ActivationFunction* act = layer_info.second;
            layers.push_back(std::make_unique<NeuralNetworkLayer>(prev_size, size, act));
            prev_size = size;
        }
    }

    void feed_forward(const DoubleVector& input, DoubleVector& output) const {
        if (input.n() != input_size) {
            throw std::runtime_error("Invalid input size");
        }
        std::vector<DoubleVector> z_values(layers.size());
        std::vector<DoubleVector> activations(layers.size() + 1);
        activations[0] = input;
        for (unsigned i = 0; i < layers.size(); ++i) {
            layers[i]->feed_forward(activations[i], z_values[i], activations[i + 1]);
        }
        output = activations.back();
    }

    double cost(const DoubleVector& input, const DoubleVector& target_output) const {
        DoubleVector output;
        feed_forward(input, output);
        double sum = 0.0;
        for (unsigned i = 0; i < output.n(); ++i) {
            double diff = output[i] - target_output[i];
            sum += diff * diff;
        }
        return 0.5 * sum;
    }

    double cost_for_training_data(const std::vector<std::pair<DoubleVector, DoubleVector>>& training_data) const {
        double total_cost = 0.0;
        for (const auto& data : training_data) {
            total_cost += cost(data.first, data.second);
        }
        return total_cost / training_data.size();
    }

    

    void train(const std::vector<std::pair<DoubleVector, DoubleVector>>& training_data,
               const double& learning_rate, const double& tol_training, const unsigned& max_iter,
               const std::string& convergence_history_file_name = "") {
        std::ofstream history_file;
        if (!convergence_history_file_name.empty()) {
            history_file.open(convergence_history_file_name);
            if (!history_file) {
                throw std::runtime_error("Cannot open history file");
            }
        }

        for (unsigned iter = 0; iter < max_iter; ++iter) {
            unsigned idx = RandomNumber::Random_number_generator() % training_data.size();
            const auto& input = training_data[idx].first;
            const auto& target = training_data[idx].second;

            // Forward pass: compute activations and z values for each layer
            std::vector<DoubleVector> z_values(layers.size());
            std::vector<DoubleVector> activations(layers.size() + 1);
            activations[0] = input;

            for (unsigned l = 0; l < layers.size(); ++l) {
                layers[l]->feed_forward(activations[l], z_values[l], activations[l + 1]);
            }

            // Backward pass: compute gradients
            std::vector<DoubleVector> delta(layers.size());
            
            // Compute output layer error
            delta.back().resize(layers.back()->get_biases().n());
            for (unsigned j = 0; j < delta.back().n(); ++j) {
                double activation = activations.back()[j];
                double dsigma = layers.back()->get_activation()->dsigma(z_values.back()[j]);
                delta.back()[j] = dsigma * (activation - target[j]);
            }

            // Backpropagate the error
            for (int l = layers.size() - 2; l >= 0; --l) {
                delta[l].resize(layers[l]->get_biases().n());
                const auto& next_weights = layers[l + 1]->get_weights();
                
                for (unsigned j = 0; j < delta[l].n(); ++j) {
                    double sum = 0.0;
                    for (unsigned k = 0; k < delta[l + 1].n(); ++k) {
                        sum += next_weights(k, j) * delta[l + 1][k];
                    }
                    
                    double dsigma = layers[l]->get_activation()->dsigma(z_values[l][j]);
                    delta[l][j] = dsigma * sum;
                }
            }

            // Update weights and biases
            for (unsigned l = 0; l < layers.size(); ++l) {
                auto& weights = const_cast<DoubleMatrix&>(layers[l]->get_weights());
                auto& biases = const_cast<DoubleVector&>(layers[l]->get_biases());

                for (unsigned j = 0; j < delta[l].n(); ++j) {
                    // Update biases
                    biases[j] -= learning_rate * delta[l][j];

                    // Update weights
                    for (unsigned k = 0; k < (l == 0 ? input_size : layers[l-1]->get_biases().n()); ++k) {
                        double input_val = (l == 0) ? input[k] : activations[l][k];
                        weights(j, k) -= learning_rate * delta[l][j] * input_val;
                    }
                }
            }

            // Periodically check convergence
            if (iter % 1000 == 0) {
                double current_cost = cost_for_training_data(training_data);
                if (!convergence_history_file_name.empty()) {
                    history_file << iter << " " << current_cost << "\n";
                }
                
                std::cout << "Iteration " << iter << ": Cost = " << current_cost << std::endl;

                if (current_cost < tol_training) {
                    break;
                }
            }
        }
    }

    void initialise_parameters(const double& mean, const double& std_dev) {
        for (auto& layer : layers) {
            layer->initialise_parameters(mean, std_dev);
        }
    }

    void write_parameters_to_disk(const std::string& filename) const {
        std::ofstream file(filename);
        if (!file) {
            throw std::runtime_error("Cannot open file for writing");
        }

        for (const auto& layer : layers) {
            layer->write_parameters(file);
        }
    }

    void read_parameters_from_disk(const std::string& filename) {
        std::ifstream file(filename);
        if (!file) {
            throw std::runtime_error("Cannot open file for reading");
        }

        for (auto& layer : layers) {
            // Read activation function name
            std::string activation_name;
            std::getline(file, activation_name);

            // Verify activation function matches
            if (activation_name != layer->get_activation()->name()) {
                throw std::runtime_error("Activation function mismatch");
            }

            // Read input and output dimensions
            unsigned input_dim, output_dim;
            file >> input_dim >> output_dim;

            // Sanity checks
            if (input_dim != layer->get_weights().m() || output_dim != layer->get_weights().n()) {
                throw std::runtime_error("Layer dimension mismatch");
            }

            // Read biases
            auto& biases = const_cast<DoubleVector&>(layer->get_biases());
            for (unsigned i = 0; i < output_dim; ++i) {
                unsigned bias_idx;
                double bias_val;
                file >> bias_idx >> bias_val;
                if (bias_idx != i) {
                    throw std::runtime_error("Bias index mismatch");
                }
                biases[i] = bias_val;
            }

            // Read weights
            auto& weights = const_cast<DoubleMatrix&>(layer->get_weights());
            for (unsigned i = 0; i < output_dim; ++i) {
                for (unsigned j = 0; j < input_dim; ++j) {
                    unsigned weight_i, weight_j;
                    double weight_val;
                    file >> weight_i >> weight_j >> weight_val;

                    if (weight_i != i || weight_j != j) {
                        throw std::runtime_error("Weight index mismatch");
                    }
                    weights(i, j) = weight_val;
                }
            }
        }
    }
    
// In project2_a.h, inside the NeuralNetwork class
static void read_training_data(const std::string& filename, 
                               std::vector<std::pair<DoubleVector, DoubleVector>>& training_data) {
    std::ifstream file(filename);
    if (!file) {
        throw std::runtime_error("Cannot open training data file: " + filename);
    }

    training_data.clear();
    double input1, input2, output;
    while (file >> input1 >> input2 >> output) {
        DoubleVector input(2);
        input[0] = input1;
        input[1] = input2;
        
        DoubleVector target(1);
        target[0] = output;
        
        training_data.push_back({input, target});
    }

    if (training_data.empty()) {
        throw std::runtime_error("No training data found in file");
    }
}

static void output_training_data(const std::string& filename, 
                                 const std::vector<std::pair<DoubleVector, DoubleVector>>& training_data) {
    std::ofstream file(filename);
    if (!file) {
        throw std::runtime_error("Cannot open output file: " + filename);
    }

    for (const auto& data_point : training_data) {
        const DoubleVector& input = data_point.first;
        const DoubleVector& output = data_point.second;
        
        file << input[0] << " " << input[1] << " " << output[0] << "\n";
    }
}
// public section of the NeuralNetwork class
void generate_decision_boundary_data() {
    std::ofstream grid_output_file("network_grid_output.txt");
    
    // Create a 100x100 grid
    for (double x1 = 0.0; x1 <= 1.0; x1 += 0.01) {
        for (double x2 = 0.0; x2 <= 1.0; x2 += 0.01) {
            // Create input vector
            DoubleVector input(2);
            input[0] = x1;
            input[1] = x2;
            
            // Get network output
            DoubleVector output;
            feed_forward(input, output);
            
            // Write to file
            grid_output_file << output[0] << " ";
        }
        grid_output_file << std::endl;
    }
    
    grid_output_file.close();
    std::cout << "Decision boundary data generated." << std::endl;
}

void compare_multiple_training_runs() {
    // Read actual training data
    std::vector<std::pair<DoubleVector, DoubleVector>> training_data;
    NeuralNetwork::read_training_data("project_training_data.dat", training_data);

    // Prepare to store results of multiple runs
    std::vector<double> initial_costs;
    std::vector<double> final_costs;
    std::vector<std::string> param_filenames;

    // Run 4 training sessions
    for (int run = 1; run <= 4; ++run) {
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

        // Compute initial cost
        double initial_cost = network.cost_for_training_data(training_data);
        initial_costs.push_back(initial_cost);

        // Prepare convergence history filename
        std::string conv_history_file = "convergence_history_run" + std::to_string(run) + ".txt";
        
        // Prepare parameter filename
        std::string param_filename = "trained_network_params_run" + std::to_string(run) + ".txt";

        // Train network and log convergence history
        network.train(training_data, 0.1, 1e-4, 10000, conv_history_file);
        
        // Compute final cost
        double final_cost = network.cost_for_training_data(training_data);
        final_costs.push_back(final_cost);

        // Save trained network parameters
        network.write_parameters_to_disk(param_filename);

        // Generate decision boundary data for each run
        std::string grid_output_file = "network_grid_output_run" + std::to_string(run) + ".txt";
        
        // Modify generate_decision_boundary_data to accept a filename
        std::ofstream grid_output_stream(grid_output_file);
        for (double x1 = 0.0; x1 <= 1.0; x1 += 0.01) {
            for (double x2 = 0.0; x2 <= 1.0; x2 += 0.01) {
                // Create input vector
                DoubleVector input(2);
                input[0] = x1;
                input[1] = x2;
                
                // Get network output
                DoubleVector output;
                network.feed_forward(input, output);
                
                // Write to file
                grid_output_stream << output[0] << " ";
            }
            grid_output_stream << std::endl;
        }
        grid_output_stream.close();

        // Store parameter filename
        param_filenames.push_back(param_filename);

        // Clean up
        delete activation_function;
    }

    // Print comparison results
    std::cout << "\nMultiple Training Runs Comparison:\n";
    std::cout << "-------------------------------\n";
    for (int i = 0; i < 4; ++i) {
        std::cout << "Run " << (i+1) << ":\n";
        std::cout << "  Initial Cost: " << initial_costs[i] << "\n";
        std::cout << "  Final Cost: " << final_costs[i] << "\n";
        std::cout << "  Cost Reduction: " 
                  << ((initial_costs[i] - final_costs[i]) / initial_costs[i] * 100.0) << "%\n";
        std::cout << "  Trained Parameters File: " << param_filenames[i] << "\n\n";
    }
}

private:
    unsigned input_size;
    std::vector<std::unique_ptr<NeuralNetworkLayer>> layers;
};

#endif // PROJECT2_A_H