# make your own neural network

import numpy as np
import scipy.special
import scipy.misc


# neural network class definition
class neuralNetwork:
    # initialize the neural network
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # set number of nodes in each input, hidden, output layer
        self.inodes = input_nodes
        self.hnodes = hidden_nodes
        self.onodes = output_nodes

        # learning rate
        self.lr = learning_rate

        # link weight matrices, wih and who
        # weights inside the arrays are w_i_j, where link is from node
        # i to node j in the next layer
        # w11 w21
        # w12 w22 etc
        self.wih = np.random.normal(0,
                                    pow(self.hnodes, -0.5),
                                    (self.hnodes, self.inodes))
        self.who = np.random.normal(0,
                                    pow(self.onodes, -0.5),
                                    (self.onodes, self.hnodes))

    # activation function
    @staticmethod
    def activation(x):
        return scipy.special.expit(x)

    # train the neural network
    def train(self, inputs_list, targets_list):
        # convert inputs list to 2d array
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = np.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = neuralNetwork.activation(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = np.dot(self.who, hidden_outputs)
        # calculate the signals emerging from finals output layer
        final_outputs = neuralNetwork.activation(final_inputs)

        # error is the (target - actual)
        output_errors = targets - final_outputs

        # hidden layer error is the output_errors, split by weight,
        # recombined at hidden nodes
        hidden_errors = np.dot(self.who.T, output_errors)

        # update the weights for the links between the hidden and output layers
        self.who += self.lr * np.dot(
            (output_errors * final_outputs * (1 - final_outputs)),
            hidden_outputs.T)

        # update the weights for the links between the input and hieedn layers
        self.wih += self.lr * np.dot(
            (hidden_errors * hidden_outputs * (1 - hidden_outputs)),
            inputs.T)

    # query the neural network
    def query(self, inputs_list):
        # convert inputs list to 2d array
        inputs = np.array(inputs_list, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = np.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = neuralNetwork.activation(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = np.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = neuralNetwork.activation(final_inputs)

        return final_outputs


if __name__ == '__main__':
    # number of input, hidden and output nodes
    input_nodes = 784
    hidden_nodes = 200
    output_nodes = 10

    # learning rate
    learning_rate = 0.1

    # create instance of neural network
    n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

    # load the mnist training data CSV file into a list
    training_data_file = open('mnist_dataset/mnist_train_100.csv', 'r')
    training_data_list = training_data_file.readlines()
    training_data_file.close()

    # train the neural network
    # go through all records in the training data set
    #
    # number of times the training data set is used for training
    # 5 times is best
    epochs = 1
    for e in range(epochs):
        for record in training_data_list:
            # create the input values
            all_values = record.split(',')
            inputs = np.asfarray(all_values[1:]) / 255.0 * 0.99 + 0.01

            # create the target values
            targets = np.zeros(output_nodes) + 0.01
            targets[int(all_values[0])] = 0.99
            n.train(inputs, targets)

    # test the neural network
    test_data_file = open('mnist_dataset/mnist_test_10.csv', 'r')
    test_data_list = test_data_file.readlines()
    test_data_file.close()

    # scorecard for how well the neural network is, initially empty
    scorecard = []

    # go through all the records in the test data set
    for record in test_data_list:
        all_values = record.split(',')
        correct_label = int(all_values[0])
        print('correct label', correct_label)

        # scale and shift the inputs
        inputs = np.asfarray(all_values[1:]) / 255.0 * 0.99 + 0.01

        # query the neural network
        outputs = n.query(inputs)

        # the index of the highest value corresponds to the label
        label = np.argmax(outputs)
        print("neural network's answer", label)

        if label == correct_label:
            scorecard.append(1)
        else:
            scorecard.append(0)

    # calculate the performance score
    print("performance = ", sum(scorecard) / len(scorecard))

    # test your own handwriting
    img_array = scipy.misc.imread('handwriting/five.png', flatten=True)
    img_data = 255.0 - img_array.reshape(784)
    img_data = img_data / 255.0 * 0.99 + 0.01
    outputs = n.query(img_data)
    label = np.argmax(outputs)
    print("neural network's answer", label)
