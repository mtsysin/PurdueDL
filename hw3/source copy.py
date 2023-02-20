from ComputationalGraphPrimer import *
import random
import operator
import numpy as np
import matplotlib.pyplot as plt
import warnings
import math
import torch
import os

seed = 0
random.seed(seed)
np.random.seed(seed)
# torch.manual_seed(seed)
# os.environ['PYTHONHASHSEED'] = str(seed)

class ComputationalGraphPrimerModified(ComputationalGraphPrimer):

    def __init__(self, *args, **kwargs):
        """Initialize the new class instance in the same way as the base class"""
        if 'modification' not in kwargs:
            warnings.warn('You are using modified CGP without specifying the modification type. \
Proceed with caution, defaults to "no". \
Supported modifications: "no", "plus", "adam"')
            self.modification = "no"
        else:
            self.modification = kwargs.pop('modification')

        if self.modification == "plus":
            if 'mu' not in kwargs:
                raise ValueError('You are trying to use "plus" modification without "mu" argument, terminate execution')
            else:
                self.mu = kwargs.pop('mu')

        if self.modification == "adam":
            if 'beta1' not in kwargs or 'beta2' not in kwargs or 'epsilon' not in kwargs:
                raise ValueError('You are trying to use "adam" modification without one of the required arguments, terminate execution')
            else:
                self.beta1 = kwargs.pop('beta1')
                self.beta2 = kwargs.pop('beta2')
                self.epsilon = kwargs.pop('epsilon')

        # Parse the rest of the input
        super().__init__(*args, **kwargs)

    ######################################################################################################
    ######################################### one neuron model ###########################################
    ######################################################################################################
    def run_training_loop_one_neuron_model(self, training_data):
        """
        The training loop must first initialize the learnable parameters.  Remember, these are the 
        symbolic names in your input expressions for the neural layer that do not begin with the 
        letter 'x'.  In this case, we are initializing with random numbers from a uniform distribution 
        over the interval (0,1).

        UPDATE: We want to override this function to be able to support SGD+ (with momentum). The original code
        form the primer is copied fully here for completeness. The things added in this homework are
        designated with the following comment line:
                                <<<<<<<<<<<<<<<<<<<<<< NEW FOR HW3
        """
        self.vals_for_learnable_params = {param: random.uniform(0,1) for param in self.learnable_params}

        self.bias = random.uniform(0,1)                   ## Adding the bias improves class discrimination.
                                                          ##   We initialize it to a random number.

        class DataLoader:
            """
            To understand the logic of the dataloader, it would help if you first understand how 
            the training dataset is created.  Search for the following function in this file:

                             gen_training_data(self)
           
            As you will see in the implementation code for this method, the training dataset
            consists of a Python dict with two keys, 0 and 1, the former points to a list of 
            all Class 0 samples and the latter to a list of all Class 1 samples.  In each list,
            the data samples are drawn from a multi-dimensional Gaussian distribution.  The two
            classes have different means and variances.  The dimensionality of each data sample
            is set by the number of nodes in the input layer of the neural network.

            The data loader's job is to construct a batch of samples drawn randomly from the two
            lists mentioned above.  And it mush also associate the class label with each sample
            separately.
            """
            def __init__(self, training_data, batch_size):
                self.training_data = training_data
                self.batch_size = batch_size
                self.class_0_samples = [(item, 0) for item in self.training_data[0]]   ## Associate label 0 with each sample
                self.class_1_samples = [(item, 1) for item in self.training_data[1]]   ## Associate label 1 with each sample

            def __len__(self):
                return len(self.training_data[0]) + len(self.training_data[1])

            def _getitem(self):    
                cointoss = random.choice([0,1])                            ## When a batch is created by getbatch(), we want the
                                                                           ##   samples to be chosen randomly from the two lists
                if cointoss == 0:
                    return random.choice(self.class_0_samples)
                else:
                    return random.choice(self.class_1_samples)            

            def getbatch(self):
                batch_data,batch_labels = [],[]                            ## First list for samples, the second for labels
                maxval = 0.0                                               ## For approximate batch data normalization
                for _ in range(self.batch_size):
                    item = self._getitem()
                    if np.max(item[0]) > maxval: 
                        maxval = np.max(item[0])
                    batch_data.append(item[0])
                    batch_labels.append(item[1])
                batch_data = [item/maxval for item in batch_data]          ## Normalize batch data
                batch = [batch_data, batch_labels]
                return batch                

##########################################################################################################################
        # We need to create a storage for previous steps if we use modification                                                     <<<<<<<<<<<<<<<<<<<<<< NEW FOR HW3   
        if self.modification == "plus":
            self.param_gradients = {param: 0 for param in self.vals_for_learnable_params}
            self.bias_gradient = 0
        if self.modification == "adam":
            # Need to add values for mk, vk 
            self.m = {param: 0 for param in self.vals_for_learnable_params}
            self.v = {param: 0 for param in self.vals_for_learnable_params}
            self.beta1_pow_k = 1                    # Values of betas to the power of k
            self.beta2_pow_k = 1   
            self.m_bias = 0   
            self.v_bias = 0              
##########################################################################################################################


        data_loader = DataLoader(training_data, batch_size=self.batch_size)
        loss_running_record = []
        i = 0
        avg_loss_over_iterations = 0.0                                    ##  Average the loss over iterations for printing out 
                                                                           ##    every N iterations during the training loop.
        for i in range(self.training_iterations):
            data = data_loader.getbatch()
            data_tuples = data[0]
            class_labels = data[1]
            y_preds, deriv_sigmoids =  self.forward_prop_one_neuron_model(data_tuples)              ##  FORWARD PROP of data
            loss = sum([(abs(class_labels[i] - y_preds[i]))**2 for i in range(len(class_labels))])  ##  Find loss
            loss_avg = loss / float(len(class_labels))                                              ##  Average the loss over batch
            avg_loss_over_iterations += loss_avg                          
            if i%(self.display_loss_how_often) == 0: 
                avg_loss_over_iterations /= self.display_loss_how_often
                loss_running_record.append(avg_loss_over_iterations)
                print("[iter=%d]  loss = %.4f" %  (i+1, avg_loss_over_iterations))                 ## Display average loss
                avg_loss_over_iterations = 0.0                                                     ## Re-initialize avg loss
            y_errors = list(map(operator.sub, class_labels, y_preds))
            y_error_avg = sum(y_errors) / float(len(class_labels))
            deriv_sigmoid_avg = sum(deriv_sigmoids) / float(len(class_labels))
            data_tuple_avg = [sum(x) for x in zip(*data_tuples)]
            data_tuple_avg = list(map(operator.truediv, data_tuple_avg, 
                                     [float(len(class_labels))] * len(class_labels) ))
            self.backprop_and_update_params_one_neuron_model(y_error_avg, data_tuple_avg, deriv_sigmoid_avg)     ## BACKPROP loss
        # plt.figure()     
        # plt.plot(loss_running_record) 
        # plt.show()   
        return loss_running_record

    def backprop_and_update_params_one_neuron_model(self, y_error, vals_for_input_vars, deriv_sigmoid):
        """
        As should be evident from the syntax used in the following call to backprop function,

           self.backprop_and_update_params_one_neuron_model( y_error_avg, data_tuple_avg, deriv_sigmoid_avg)
                                                                     ^^^             ^^^                ^^^
        the values fed to the backprop function for its three arguments are averaged over the training 
        samples in the batch.  This in keeping with the spirit of SGD that calls for averaging the 
        information retained in the forward propagation over the samples in a batch.

        See Slide 59 of my Week 3 slides for the math of back propagation for the One-Neuron network.
        """
        input_vars = self.independent_vars
        input_vars_to_param_map = self.var_to_var_param[self.output_vars[0]]
        param_to_vars_map = {param : var for var, param in input_vars_to_param_map.items()}
        vals_for_input_vars_dict =  dict(zip(input_vars, list(vals_for_input_vars)))
##########################################################################################################################
        # vals_for_learnable_params = self.vals_for_learnable_params                                             <<<<<<<<<<<<<<<<<<<<<< NEW FOR HW3  
        if self.modification == "no":                                                   # Keep original code
            for i, param in enumerate(self.vals_for_learnable_params):
                ## Calculate the next step in the parameter hyperplane
    #            step = self.learning_rate * y_error * vals_for_input_vars_dict[input_vars[i]] * deriv_sigmoid    
                step = self.learning_rate * y_error * vals_for_input_vars_dict[param_to_vars_map[param]] * deriv_sigmoid    
                ## Update the learnable parameters
                self.vals_for_learnable_params[param] += step
            self.bias += self.learning_rate * y_error * deriv_sigmoid    ## Update the bias


        elif self.modification == "plus":                                               # Logic for "plus" modification
            for param in self.vals_for_learnable_params:
                # Calculate new grad from old grads   
                gradient = (y_error * vals_for_input_vars_dict[param_to_vars_map[param]] * deriv_sigmoid +  # New part (essentially negative gradient)
                            self.mu * self.param_gradients[param])                                          # Old part
                # Update the learnable parameters
                self.vals_for_learnable_params[param] += self.learning_rate * gradient
                # Save for future calculations
                self.param_gradients[param] = gradient
            gradient = y_error * deriv_sigmoid + \
                        self.mu * self.bias_gradient                   # Negative gradient for bias
            # Update bias term
            self.bias += self.learning_rate * gradient    ## Update the bias
            self.bias_gradient = gradient                   # Update bias gradient.


        elif self.modification == "adam":
            self.beta1_pow_k *= self.beta1                                                # Update powers of betas
            self.beta2_pow_k *= self.beta2 
            for param in self.vals_for_learnable_params:
                # Calculate new grad from old grads   
                m = ((1 - self.beta1) * y_error * vals_for_input_vars_dict[param_to_vars_map[param]] * deriv_sigmoid +  # New part (essentially negative gradient)
                            self.beta1 * self.m[param])
                v = ((1 - self.beta2) * (y_error * vals_for_input_vars_dict[param_to_vars_map[param]] * deriv_sigmoid) ** 2 +  # New part (essentially negative gradient)
                            self.beta2 * self.v[param])                                                                         # Old part

                m_hat = m / (1 - self.beta1_pow_k)
                v_hat = v / (1 - self.beta2_pow_k)

                # Update the learnable parameters
                self.vals_for_learnable_params[param] += self.learning_rate * m_hat / math.sqrt(v_hat + self.epsilon)
                # Save for future calculations
                self.m[param] = m
                self.v[param] = v

            m_bias = ((1 - self.beta1) * y_error * deriv_sigmoid + 
                        self.beta1 * self.m_bias)                    # Negative gradient for bias
            v_bias = ((1 - self.beta2) * (y_error * deriv_sigmoid)**2 + 
                        self.beta2 * self.v_bias)                    # Negative gradient for bia
            m_bias_hat = m_bias / (1 - self.beta1_pow_k)
            v_bias_hat = v_bias / (1 - self.beta2_pow_k)   
            
            # Update bias term
            self.bias += self.learning_rate * m_bias_hat / math.sqrt(v_bias_hat + self.epsilon)   ## Update the bias
            self.m_bias = m_bias
            self.v_bias = v_bias
        else:
            raise ValueError("Wrong modification supplied")
    ######################################################################################################


    ######################################################################################################
    ######################################## multi neuron model ##########################################
    ######################################################################################################

    def run_training_loop_multi_neuron_model(self, training_data):

        class DataLoader:
            """
            To understand the logic of the dataloader, it would help if you first understand how 
            the training dataset is created.  Search for the following function in this file:

                             gen_training_data(self)
           
            As you will see in the implementation code for this method, the training dataset
            consists of a Python dict with two keys, 0 and 1, the former points to a list of 
            all Class 0 samples and the latter to a list of all Class 1 samples.  In each list,
            the data samples are drawn from a multi-dimensional Gaussian distribution.  The two
            classes have different means and variances.  The dimensionality of each data sample
            is set by the number of nodes in the input layer of the neural network.

            The data loader's job is to construct a batch of samples drawn randomly from the two
            lists mentioned above.  And it mush also associate the class label with each sample
            separately.
            """
            def __init__(self, training_data, batch_size):
                self.training_data = training_data
                self.batch_size = batch_size
                self.class_0_samples = [(item, 0) for item in self.training_data[0]]    ## Associate label 0 with each sample
                self.class_1_samples = [(item, 1) for item in self.training_data[1]]    ## Associate label 1 with each sample

            def __len__(self):
                return len(self.training_data[0]) + len(self.training_data[1])

            def _getitem(self):    
                cointoss = random.choice([0,1])                            ## When a batch is created by getbatch(), we want the
                                                                           ##   samples to be chosen randomly from the two lists
                if cointoss == 0:
                    return random.choice(self.class_0_samples)
                else:
                    return random.choice(self.class_1_samples)            

            def getbatch(self):
                batch_data,batch_labels = [],[]                            ## First list for samples, the second for labels
                maxval = 0.0                                               ## For approximate batch data normalization
                for _ in range(self.batch_size):
                    item = self._getitem()
                    if np.max(item[0]) > maxval: 
                        maxval = np.max(item[0])
                    batch_data.append(item[0])
                    batch_labels.append(item[1])
                batch_data = [item/maxval for item in batch_data]          ## Normalize batch data       
                batch = [batch_data, batch_labels]
                return batch                


        """
        The training loop must first initialize the learnable parameters.  Remember, these are the 
        symbolic names in your input expressions for the neural layer that do not begin with the 
        letter 'x'.  In this case, we are initializing with random numbers from a uniform distribution 
        over the interval (0,1).
        """
        self.vals_for_learnable_params = {param: random.uniform(0,1) for param in self.learnable_params}

        self.bias = [random.uniform(0,1) for _ in range(self.num_layers-1)]      ## Adding the bias to each layer improves 
                                                                                 ##   class discrimination. We initialize it 
                                                                                 ##   to a random number.

##########################################################################################################################
        # We need to create a storage for previous steps if we use modification                                                     <<<<<<<<<<<<<<<<<<<<<< NEW FOR HW3   
        if self.modification == "plus":
            self.param_gradients = {param: 0 for param in self.vals_for_learnable_params}
            self.bias_gradients = [0 for _ in range(self.num_layers-1)]
        if self.modification == "adam":
            # Need to add values for mk, vk, 
            self.m = {param: 0 for param in self.vals_for_learnable_params}
            self.v = {param: 0 for param in self.vals_for_learnable_params}
            self.beta1_pow_k = 1                    # Values of betas to the power of k
            self.beta2_pow_k = 1   
            self.m_bias = [0 for _ in range(self.num_layers-1)]  
            self.v_bias = [0 for _ in range(self.num_layers-1)]             
           
##########################################################################################################################

        data_loader = DataLoader(training_data, batch_size=self.batch_size)
        loss_running_record = []
        i = 0
        avg_loss_over_iterations = 0.0                                          ##  Average the loss over iterations for printing out 
                                                                                 ##    every N iterations during the training loop.   
        for i in range(self.training_iterations):
            data = data_loader.getbatch()
            data_tuples = data[0]
            class_labels = data[1]
            self.forward_prop_multi_neuron_model(data_tuples)                                  ## FORW PROP works by side-effect 
            predicted_labels_for_batch = self.forw_prop_vals_at_layers[self.num_layers-1]      ## Predictions from FORW PROP
            y_preds =  [item for sublist in  predicted_labels_for_batch  for item in sublist]  ## Get numeric vals for predictions
            loss = sum([(abs(class_labels[i] - y_preds[i]))**2 for i in range(len(class_labels))])  ## Calculate loss for batch
            loss_avg = loss / float(len(class_labels))                                         ## Average the loss over batch
            avg_loss_over_iterations += loss_avg                                              ## Add to Average loss over iterations
            if i%(self.display_loss_how_often) == 0: 
                avg_loss_over_iterations /= self.display_loss_how_often
                loss_running_record.append(avg_loss_over_iterations)
                print("[iter=%d]  loss = %.4f" %  (i+1, avg_loss_over_iterations))            ## Display avg loss
                avg_loss_over_iterations = 0.0                                                ## Re-initialize avg-over-iterations loss
            y_errors = list(map(operator.sub, class_labels, y_preds))
            y_error_avg = sum(y_errors) / float(len(class_labels))
            self.backprop_and_update_params_multi_neuron_model(y_error_avg, class_labels)      ## BACKPROP loss
        # plt.figure()     
        # plt.plot(loss_running_record) 
        # plt.show()   
        return loss_running_record



    def backprop_and_update_params_multi_neuron_model(self, y_error, class_labels):
        """
        First note that loop index variable 'back_layer_index' starts with the index of
        the last layer.  For the 3-layer example shown for 'forward', back_layer_index
        starts with a value of 2, its next value is 1, and that's it.

        Stochastic Gradient Gradient calls for the backpropagated loss to be averaged over
        the samples in a batch.  To explain how this averaging is carried out by the
        backprop function, consider the last node on the example shown in the forward()
        function above.  Standing at the node, we look at the 'input' values stored in the
        variable "input_vals".  Assuming a batch size of 8, this will be list of
        lists. Each of the inner lists will have two values for the two nodes in the
        hidden layer. And there will be 8 of these for the 8 elements of the batch.  We average
        these values 'input vals' and store those in the variable "input_vals_avg".  Next we
        must carry out the same batch-based averaging for the partial derivatives stored in the
        variable "deriv_sigmoid".

        Pay attention to the variable 'vars_in_layer'.  These store the node variables in
        the current layer during backpropagation.  Since back_layer_index starts with a
        value of 2, the variable 'vars_in_layer' will have just the single node for the
        example shown for forward(). With respect to what is stored in vars_in_layer', the
        variables stored in 'input_vars_to_layer' correspond to the input layer with
        respect to the current layer. 
        """
        # backproped prediction error:
        pred_err_backproped_at_layers = {i : [] for i in range(1,self.num_layers-1)}  
        pred_err_backproped_at_layers[self.num_layers-1] = [y_error]
        for back_layer_index in reversed(range(1,self.num_layers)):
            input_vals = self.forw_prop_vals_at_layers[back_layer_index -1]
            input_vals_avg = [sum(x) for x in zip(*input_vals)]
            input_vals_avg = list(map(operator.truediv, input_vals_avg, [float(len(class_labels))] * len(class_labels)))
            deriv_sigmoid =  self.gradient_vals_for_layers[back_layer_index]
            deriv_sigmoid_avg = [sum(x) for x in zip(*deriv_sigmoid)]
            deriv_sigmoid_avg = list(map(operator.truediv, deriv_sigmoid_avg, 
                                                             [float(len(class_labels))] * len(class_labels)))
            vars_in_layer  =  self.layer_vars[back_layer_index]                 ## a list like ['xo']
            vars_in_next_layer_back  =  self.layer_vars[back_layer_index - 1]   ## a list like ['xw', 'xz']

            layer_params = self.layer_params[back_layer_index]         
            ## note that layer_params are stored in a dict like        
                ##     {1: [['ap', 'aq', 'ar', 'as'], ['bp', 'bq', 'br', 'bs']], 2: [['cp', 'cq']]}
            ## "layer_params[idx]" is a list of lists for the link weights in layer whose output nodes are in layer "idx"
            transposed_layer_params = list(zip(*layer_params))         ## creating a transpose of the link matrix

            backproped_error = [None] * len(vars_in_next_layer_back)
            for k,varr in enumerate(vars_in_next_layer_back):
                for j,var2 in enumerate(vars_in_layer):
                    backproped_error[k] = sum([self.vals_for_learnable_params[transposed_layer_params[k][i]] * 
                                               pred_err_backproped_at_layers[back_layer_index][i] 
                                               for i in range(len(vars_in_layer))])
#                                               deriv_sigmoid_avg[i] for i in range(len(vars_in_layer))])
            pred_err_backproped_at_layers[back_layer_index - 1]  =  backproped_error
                        # vals_for_learnable_params = self.vals_for_learnable_params                                             <<<<<<<<<<<<<<<<<<<<<< NEW FOR HW3  
            if self.modification == "no":                                                   # Keep original code
                for j,var in enumerate(vars_in_layer):
                    layer_params = self.layer_params[back_layer_index][j]
                    for i,param in enumerate(layer_params):
                        gradient_of_loss_for_param = input_vals_avg[i] * pred_err_backproped_at_layers[back_layer_index][j] 
                        step = self.learning_rate * gradient_of_loss_for_param * deriv_sigmoid_avg[j] 
                        self.vals_for_learnable_params[param] += step
                self.bias[back_layer_index-1] += self.learning_rate * sum(pred_err_backproped_at_layers[back_layer_index]) \
                                                                            * sum(deriv_sigmoid_avg)/len(deriv_sigmoid_avg)


            elif self.modification == "plus":                                               # Logic for "plus" modification
                for j,var in enumerate(vars_in_layer):
                    layer_params = self.layer_params[back_layer_index][j] 
                    for i,param in enumerate(layer_params):

                        gradient_of_loss_for_param = input_vals_avg[i] * pred_err_backproped_at_layers[back_layer_index][j] * deriv_sigmoid_avg[j] \
                                                    + self.mu * self.param_gradients[param]
                        self.vals_for_learnable_params[param] += self.learning_rate * gradient_of_loss_for_param
                        self.param_gradients[param] = gradient_of_loss_for_param

                gradient_of_loss_for_bias = sum(pred_err_backproped_at_layers[back_layer_index]) * sum(deriv_sigmoid_avg)/len(deriv_sigmoid_avg) + \
                                            self.mu * self.bias_gradients[back_layer_index-1]
                
                self.bias[back_layer_index-1] += self.learning_rate * gradient_of_loss_for_bias
                self.bias_gradients[back_layer_index-1] = gradient_of_loss_for_bias


            elif self.modification == "adam":
                self.beta1_pow_k *= self.beta1                                                # Update powers of betas
                self.beta2_pow_k *= self.beta2 

                for j,var in enumerate(vars_in_layer):
                    layer_params = self.layer_params[back_layer_index][j] 
                    for i,param in enumerate(layer_params):
                        m = (1 - self.beta1) * input_vals_avg[i] * pred_err_backproped_at_layers[back_layer_index][j] * deriv_sigmoid_avg[j] \
                                                    + self.beta1 * self.m[param]
                        v = (1 - self.beta2) * (input_vals_avg[i] * pred_err_backproped_at_layers[back_layer_index][j] * deriv_sigmoid_avg[j])**2 \
                                                    + self.beta2 * self.v[param]

                        m_hat = m / (1 - self.beta1_pow_k)
                        v_hat = v / (1 - self.beta2_pow_k)
                        
                        self.vals_for_learnable_params[param] += self.learning_rate * m_hat / math.sqrt(v_hat + self.epsilon)
                        self.m[param] = m
                        self.v[param] = v

                m_bias = (1 - self.beta1) * sum(pred_err_backproped_at_layers[back_layer_index]) * sum(deriv_sigmoid_avg)/len(deriv_sigmoid_avg) \
                                            + self.beta1 * self.m_bias[back_layer_index-1]
                v_bias = (1 - self.beta2) * (sum(pred_err_backproped_at_layers[back_layer_index]) * sum(deriv_sigmoid_avg)/len(deriv_sigmoid_avg))**2 \
                                            + self.beta2 * self.v_bias[back_layer_index-1]    
                m_bias_hat = m_bias / (1 - self.beta1_pow_k)
                v_bias_hat = v_bias / (1 - self.beta2_pow_k)   

                self.bias[back_layer_index-1] += self.learning_rate * m_bias_hat / math.sqrt(v_bias_hat + self.epsilon) 
                self.m_bias[back_layer_index-1] = m_bias
                self.v_bias[back_layer_index-1] = v_bias
            else:
                raise ValueError("Wrong modification supplied")

    ######################################################################################################


if __name__=="__main__":
    """
    Test code
    The first part is directly copied from the original example.
    Second part builds upon it by creating an instance of the modified class and compares their performance.
    """

    ITER_NUM = 40000
    seed = 0           
    random.seed(seed)
    np.random.seed(seed)

    def run_one(cgp):
        cgp.parse_expressions()

        # cgp.display_one_neuron_network()      

        training_data = cgp.gen_training_data()

        return cgp.run_training_loop_one_neuron_model(training_data)

    cgp = ComputationalGraphPrimerModified(
        modification = "no",
        one_neuron_model = True,
        expressions = ['xw=ab*xa+bc*xb+cd*xc+ac*xd'],
        output_vars = ['xw'],
        dataset_size = 5000,
        # learning_rate = 1e-3,
              learning_rate = 5 * 1e-2,
        training_iterations = ITER_NUM,
        batch_size = 8,
        display_loss_how_often = 100,
        debug = True,
    )

    # no = run_one(cgp)

    # Create an instance on a modified class (sgd +):
    cgp = ComputationalGraphPrimerModified(
        modification = "plus",
        mu = 0.95,
        one_neuron_model = True,
        expressions = ['xw=ab*xa+bc*xb+cd*xc+ac*xd'],
        output_vars = ['xw'],
        dataset_size = 5000,
        # learning_rate = 1e-3,
              learning_rate = 5 * 1e-2,
        training_iterations = ITER_NUM,
        batch_size = 8,
        display_loss_how_often = 100,
        debug = True,
    )

    # plus = run_one(cgp)

    # Create an instance on a modified class (adam):
    cgp = ComputationalGraphPrimerModified(
        modification = "adam",
        beta1 = 0.9,
        beta2 = 0.99,
        epsilon = 0.00000001,
        one_neuron_model = True,
        expressions = ['xw=ab*xa+bc*xb+cd*xc+ac*xd'],
        output_vars = ['xw'],
        dataset_size = 5000,
        # learning_rate = 1e-3,
              learning_rate = 5 * 1e-2,
        training_iterations = ITER_NUM,
        batch_size = 8,
        display_loss_how_often = 100,
        debug = True,
    )

    # adam = run_one(cgp)

    # plt.figure()     
    # plt.plot(no, label="SGD") 
    # plt.plot(plus, label="SGD+") 
    # plt.plot(adam, label="ADAM") 
    # plt.legend()

    # plt.show()  


    ######################################################################################################

    def run_multi(cgp):
        cgp.parse_multi_layer_expressions()
        # cgp.display_multi_neuron_network()   
        training_data = cgp.gen_training_data()

        return cgp.run_training_loop_multi_neuron_model( training_data )    

    cgp = ComputationalGraphPrimerModified(
        modification = "no",
        num_layers = 3,
        layers_config = [4,2,1],                         # num of nodes in each layer
        expressions = ['xw=ap*xp+aq*xq+ar*xr+as*xs',
                        'xz=bp*xp+bq*xq+br*xr+bs*xs',
                        'xo=cp*xw+cq*xz'],
        output_vars = ['xo'],
        dataset_size = 5000,
        learning_rate = 1e-3,
                #   learning_rate = 5 * 1e-3,
        training_iterations = ITER_NUM,
        batch_size = 8,
        display_loss_how_often = 100,
        debug = True,
    )

    no = run_multi(cgp)

    cgp = ComputationalGraphPrimerModified(
        modification = "plus",
        mu = 0.99,
        num_layers = 3,
        layers_config = [4,2,1],                         # num of nodes in each layer
        expressions = ['xw=ap*xp+aq*xq+ar*xr+as*xs',
                        'xz=bp*xp+bq*xq+br*xr+bs*xs',
                        'xo=cp*xw+cq*xz'],
        output_vars = ['xo'],
        dataset_size = 5000,
        learning_rate = 1e-3,
                #   learning_rate = 5 * 1e-3,
        training_iterations = ITER_NUM,
        batch_size = 8,
        display_loss_how_often = 100,
        debug = True,
    )

    plus = run_multi(cgp)

    cgp = ComputationalGraphPrimerModified(
        modification = "adam",
        beta1 = 0.9,
        beta2 = 0.99,
        epsilon = 0.0000001,
        num_layers = 3,
        layers_config = [4,2,1],                         # num of nodes in each layer
        expressions = ['xw=ap*xp+aq*xq+ar*xr+as*xs',
                        'xz=bp*xp+bq*xq+br*xr+bs*xs',
                        'xo=cp*xw+cq*xz'],
        output_vars = ['xo'],
        dataset_size = 5000,
        learning_rate = 1e-3,
                #   learning_rate = 5 * 1e-3,
        training_iterations = ITER_NUM,
        batch_size = 8,
        display_loss_how_often = 100,
        debug = True,
    )

    adam = run_multi(cgp)

    plt.figure()     
    plt.plot(no, label="SGD") 
    plt.plot(plus, label="SGD+") 
    plt.plot(adam, label="ADAM") 
    plt.legend()
    plt.show()  





