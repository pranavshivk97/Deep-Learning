import numpy as np

class VectorComputationalGraph:
    def __init__(self, x, W):
        self.x = x
        self.W = W

    def multiply(self):
        return np.matmul(self.W, self.x)

    def sigmoid(self, op):
        return 1/(1+np.exp(-op))

    def l2_loss(self, op):
        return np.linalg.norm(op)**2

    def diff_l2(self, op):
        return np.multiply(2, op)

    def diff_sigmoid(self, op):
        return np.multiply((1 - self.sigmoid(op)), self.sigmoid(op))

    def diff_mult(self, op):
        return 2 * np.matmul(np.transpose(self.W), op), 2 * np.multiply(op, np.transpose(self.x))
    
def forward_propogation(x, W):
    graph = VectorComputationalGraph(x, W)
    f_prop = {}

    f_prop['mult'] = graph.multiply()
    f_prop['sigmoid'] = graph.sigmoid(f_prop['mult'])
    f_prop['l2'] = graph.l2_loss(f_prop['sigmoid'])

    return f_prop

def backward_propogation(f_prop, x, W):
    graph = VectorComputationalGraph(x, W)
    b_prop = {}

    b_prop['bp1'] = np.multiply(1, graph.diff_l2(f_prop['sigmoid']))
    b_prop['bp2'] = np.multiply(b_prop['bp1'], graph.diff_sigmoid(f_prop['mult']))
    b_prop['x'], b_prop['W'] = graph.diff_mult(b_prop['bp2']) 

    gradients = {'x': b_prop['x'], 'W': b_prop['W']}
    return b_prop

def main():
    W = np.array([[1, 2, 3], [3, 4, 5], [5, 4, 3]])
    x = np.array([[1], [0], [1]])

    f_prop = forward_propogation(x, W)
    print("FORWARD PROPOGATION\n", f_prop)
    gradients = backward_propogation(f_prop, x, W)
    print("\nBACKWARD PROPOGATION")
    for key, val in gradients.items():
        print(key, ":\n", val)


if __name__ == '__main__':
    main()  
