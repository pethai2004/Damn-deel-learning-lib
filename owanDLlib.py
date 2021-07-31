import numpy as np
import tensorflow as tf

# class initial_parameter():
#     def __init__(self, size=(1, 1, 1)):
#         self.size = size

#     def initialize(self):
#         initialized = np.random.rand(self.size)

#     def random_uniform(self, boundary=(-1, 1)):
#         initialized = np.random.random_uniform(boundary, self.size)
class SimpleDense():
    def __init__(self, n_inputs, n_nodes, activation=None):
        self.act = activation
        self.n_inputs = n_inputs
        self.n_nodes = n_nodes
        self.W = np.random.rand(n_nodes, n_inputs)
        self.b = np.random.rand()
        
    def Activation(self, x):
        return 1/(1+np.exp(-x))
    
    def forward(self, inputs): #inputs shape = (n, 1)
        inputs = inputs.reshape(self.n_inputs, 1)
        linear_out = np.dot(self.W, inputs) + b
        out = linear_out
        if self.act==None:
            pass
        else:
            out = self.Activation(linear_out)
        return out
    
    def backward(self, inputs, epochs):
        gradies = {}
        
        for i in range(epochs):
            pass
            
        
    def predict(self, inputs):
        o = np.dot(self.W, inputs) + b #unnormalized log prob
        print(o.shape)
        prob = np.exp(o)/np.sum(np.exp(0), keepdims=True)
        return np.argmax(prob)
    
    def build(self, inputs):
        pass
        
        
class CNNs():
    def __init__(self, ):
        
    def CNN(data, kernel, s=1, p=None, pad=False, c=0, original_dim =(None, None, 3)):
        n_kernel, h_kernel, w_kernel, chan = kernel.shape
        h_data, w_data, chan = data.shape

        h_remain = (h_data - h_kernel) % s  
        w_remain = (w_data - w_kernel) % s 
        print('remain=', h_remain, w_remain)
        if pad==False:
            if h_remain==0: 
                h = h_data
            else:
                h = h_data - h_remain
            if w_remain==0:
                w = w_data
            else:
                w = w_data - w_remain 
            data = data[:h,:w,:]
            h_output = int((h_data - h_kernel - h_remain)/s + 1)
            w_output = int((w_data - w_kernel - w_remain)/s + 1)

        elif pad==True:
            if (h_remain%2, w_remain%2)==(0,0):
                h_p = int(h_remain/2)
                w_p = int(w_remain/2)
                h_output = int((h_data - h_kernel + h_p*2)/s + 1)     
                w_output = int((w_data - w_kernel + w_p*2)/s + 1)

            else:
                raise Exception('Dame, resize in to make remainder divisible by 2')
            padded = np.zeros((h_data+2*h_p, w_data+2*h_p, chan)) # output of padded       
            for i in range(chan):
                padded[:,:,i] = np.pad(data[:,:,i] , pad_width=((h_p, h_p), (w_p, w_p)), constant_values=0)
            data = padded
        output = np.zeros((n_kernel, h_output, w_output,  chan))
        for k in range(n_kernel):
            y = 0
            for h in range(h_output):
                x = 0 
                for w in range(w_output):  

                    output[k, h, w] = np.sum(kernel[k] * data[y:y + h_kernel, x:x + w_kernel], axis=(0,1))
                x = x + s
            y = y + s   
        return output
    
    
class RNNs():
    def __init__(self, seq, vocab_size, seq_length, h_size=100):
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.h_size = h_size
        
        self.parameters = {}
        self.parameters['U'] = np.random.rand(h_size, vocab_size)
        self.parameters['W'] = np.random.rand(h_size, h_size)
        self.parameters['V'] = np.random.rand(vocab_size, h_size)
        self.parameters['b'] = np.random.rand(h_size, 1)
        self.parameters['c'] = np.random.rand(vocab_size, 1)
        self.h_t0 = np.zeros((self.h_size, 1))
        
        self.grads = {}
        self.optimizers_parameters = {}
        for k in self.parameters:
            self.grads['d'+ k] = np.zeros_like(self.parameters[k])
            self.optimizers_parameters['m' + k] = np.zeros_like(self.parameters[k])
        self.h = np.zeros((seq_length))
        self.o = np.zeros((seq_length))
        self.y = np.zeros((seq_length))
        
    def softmax(self, x):
        return np.exp(x)/ np.sum(np.exp(x))
    
    def forward(self, inputs_x, previous_h):
        hid_state, y_pred, o_state = {}, {}, {}
        hid_state[-1] = np.copy(previous_h)
        
        for t in range(len(inputs_x)):
            W_h = np.dot(self.parameters['W'], hid_state[t-1]) # 100x100 , 100,1
            U_x = np.dot(self.parameters['U'], inputs_x[t]).reshape(100,1)    # 100,8000, 8000,1
            #print('W_h=', W_h.shape,'U_x=',U_x.shape, 'W_h+U_x=',(W_h+U_x).shape)
            A = self.parameters['b'] + W_h + U_x
            #print('b=',self.parameters['U'].shape ,"A=",A.shape)
            hid_state[t] = np.tanh(A)
            V_h = np.dot(self.parameters['V'], hid_state[t])
            #print('V_h=',V_h.shape)
            o_state = self.parameters['c'] + V_h
            #print('c=',self.parameters['c'].shape,'o_state=',o_state.shape)
            y_pred = self.softmax(o_state)
            self.h_t0 = hid_state[t]
            #print('----------------------')
        return y_pred, hid_state, o_state
    
    def error(self, inputs_x, p_model, prediction):
        return -np.sum(np.log(p_model) * prediction)
    
    def backward(self):
        pass

class SimpleNN():
class BackProp():
class Optimizer():
