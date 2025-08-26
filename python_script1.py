import numpy as np
import matplotlib.pyplot as plt
import time

# Assuming you have the NeuralNetwork class from your project2_a.h implementation
# If not, you'll need to import or define it here

def read_training_data(filename):
    """
    Read training data from a file
    """
    data = []
    with open(filename, 'r') as file:
        for line in file:
            try:
                # Split line and convert to float
                values = [float(x) for x in line.strip().split()]
                if len(values) == 3:
                    data.append(values)
            except ValueError:
                continue
    return data

def assess_network_performance(network, training_data, test_data=None):
    """
    Comprehensive performance assessment
    """
    # Compute training cost
    training_cost = network.cost_for_training_data(training_data)
    
    # Compute classification accuracy
    correct_predictions = 0
    total_predictions = len(training_data)
    
    predictions = []
    true_labels = []
    
    for data_point in training_data:
        # Prepare input vector
        input_vector = DoubleVector(2)
        input_vector[0] = data_point[0]
        input_vector[1] = data_point[1]
        
        # Get network prediction
        output = DoubleVector(1)
        network.feed_forward(input_vector, output)
        
        # Convert output to binary classification
        predicted_label = 1 if output[0] >= 0.5 else -1
        true_label = data_point[2]
        
        predictions.append(predicted_label)
        true_labels.append(true_label)
        
        if predicted_label == true_label:
            correct_predictions += 1
    
    # Compute accuracy
    accuracy = correct_predictions / total_predictions * 100
    
    # Compute confusion matrix
    confusion_matrix = np.zeros((2, 2), dtype=int)
    for pred, true in zip(predictions, true_labels):
        row = 0 if true == 1 else 1
        col = 0 if pred == 1 else 1
        confusion_matrix[row, col] += 1
    
    # Performance metrics
    performance = {
        'training_cost': training_cost,
        'accuracy': accuracy,
        'confusion_matrix': confusion_matrix,
        'true_positives': confusion_matrix[0, 0],
        'false_positives': confusion_matrix[1, 0],
        'true_negatives': confusion_matrix[1, 1],
        'false_negatives': confusion_matrix[0, 1],
        'precision': (confusion_matrix[0, 0] / 
                      (confusion_matrix[0, 0] + confusion_matrix[1, 0]) 
                      if (confusion_matrix[0, 0] + confusion_matrix[1, 0]) > 0 else 0),
        'recall': (confusion_matrix[0, 0] / 
                   (confusion_matrix[0, 0] + confusion_matrix[0, 1]) 
                   if (confusion_matrix[0, 0] + confusion_matrix[0, 1]) > 0 else 0)
    }
    
    return performance

def train_and_analyze_network(architecture, training_data, max_iter=4000000):
    """
    Train a neural network and analyze its performance
    """
    # Create activation function
    act_func = TanhActivationFunction()
    
    # Create network
    network = create_neural_network(architecture, act_func)
    
    # Initialize parameters
    network.initialise_parameters(0.0, 0.1)
    
    # Prepare convergence tracking
    convergence_file = f"convergence_history_arch{'_'.join(map(str, architecture))}.txt"
    
    # Start timing
    start_time = time.time()
    
    # Train the network
    network.train(training_data, 0.01, 1e-3, max_iter, convergence_file)
    
    # Calculate training time
    training_time = time.time() - start_time
    
    # Assess network performance
    performance = assess_network_performance(network, training_data)
    
    # Generate decision boundary
    grid_output_file = f"grid_output_arch{'_'.join(map(str, architecture))}.txt"
    generate_decision_boundary(network, grid_output_file)
    
    # Return analysis results
    return {
        'architecture': architecture,
        'training_time': training_time,
        'performance': performance,
        'convergence_file': convergence_file,
        'grid_output_file': grid_output_file
    }

def main():
    # Load training data from spiral dataset
    try:
        training_data = read_training_data('spiral_training_data.dat')
    except FileNotFoundError:
        print("Spiral dataset not found. Using project training data instead.")
        training_data = read_training_data('project_training_data.dat')
    
    # Define architectures to explore
    architectures = [
        [2, 4, 4, 1],
        [2, 4, 4, 4, 1],
        [2, 4, 4, 4, 4, 1],
        [2, 4, 8, 8, 1],
        [2, 4, 16, 16, 1]
    ]
    
    # Results storage
    results = []
    
    # Train networks and collect results
    for arch in architectures:
        print(f"\nTraining network with architecture: {arch}")
        result = train_and_analyze_network(arch, training_data)
        results.append(result)
        
        # Print detailed performance
        perf = result['performance']
        print("\nPerformance Metrics:")
        print(f"Training Cost: {perf['training_cost']:.6f}")
        print(f"Accuracy: {perf['accuracy']:.2f}%")
        print(f"Precision: {perf['precision']:.4f}")
        print(f"Recall: {perf['recall']:.4f}")
        print("Confusion Matrix:")
        print(perf['confusion_matrix'])
    
    # Visualize results
    plot_convergence_histories(results)
    visualize_decision_boundaries(results, training_data)

if __name__ == "__main__":
    main()