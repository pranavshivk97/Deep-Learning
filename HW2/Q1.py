import math

class ScalarComputationalGraph:
    def __init__(self, x1, w1, x2, w2):
        self.x1 = x1
        self.w1 = w1
        self.x2 = x2
        self.w2 = w2

    # FORWARD PROPAGATION FUNCTIONS
    def multiply_nodes(self, op1, op2):
        return op1 * op2

    def sin(self, op):
        return math.sin(op)

    def cos(self, op):
        return math.cos(op)

    def square(self, op):
        return op**2

    def add_nodes(self, op1, op2):
        return op1 + op2

    def add_constant(self, op, const):
        return const + op

    def reciprocal(self, op):
        return 1/op

    # BACKWARD PROPAGATION FUNCTIONS
    def diff_reciprocal(self, op):
        return -1/(op ** 2)

    def diff_const_sum(self, op):
        return 1
    
    def diff_square(self, op):
        return 2*op

    def diff_sin(self, op):
        return math.cos(op)

    def diff_cos(self, op):
        return -math.sin(op)

    
def forward_propogation(x1, w1, x2, w2):
    graph = ScalarComputationalGraph(x1, w1, x2, w2)
    f_prop = {}

    f_prop['mult1'] = graph.multiply_nodes(x1, w1)
    f_prop['mult2'] = graph.multiply_nodes(x2, w2)
    f_prop['sin'] = graph.sin(f_prop['mult1'])
    f_prop['cos'] = graph.cos(f_prop['mult2'])
    f_prop['square'] = graph.square(f_prop['sin'])
    f_prop['sum'] = graph.add_nodes(f_prop['square'], f_prop['cos'])
    f_prop['const'] = graph.add_constant(f_prop['sum'], 2)
    f_prop['f'] = graph.reciprocal(f_prop['const'])
    
    return f_prop

def backward_propogation(f_prop, x1, w1, x2, w2):
    graph = ScalarComputationalGraph(x1, w1, x2, w2)
    b_prop = {}

    b_prop['bp1'] = 1 * graph.diff_reciprocal(f_prop['const'])
    b_prop['bp2'] = b_prop['bp1'] * graph.diff_const_sum(f_prop['sum'])
    b_prop['branch1'] = b_prop['branch2'] = b_prop['bp2'] 
    b_prop['bp5'] = b_prop['branch1'] * graph.diff_square(f_prop['sin'])
    b_prop['bp6'] = b_prop['branch2'] * graph.diff_cos(f_prop['mult2'])
    b_prop['bp7'] = b_prop['bp5'] * graph.diff_sin(f_prop['mult1'])
    b_prop['x1'], b_prop['w1'] = b_prop['bp7'] * w1, b_prop['bp7'] * x1
    b_prop['x2'], b_prop['w2'] = b_prop['bp6'] * w2, b_prop['bp6'] * x2

    gradients = {'x1': b_prop['x1'], 'w1': b_prop['w1'], 'x2': b_prop['x2'], 'w2': b_prop['w2']}

    return gradients

def main():
    x1, w1, x2, w2 = 2, -1, -3, 2
    f_prop = forward_propogation(x1, w1, x2, w2)
    print("FORWARD PROPOGATION\n", f_prop['f'])

    gradients = backward_propogation(f_prop, x1, w1, x2, w2)
    print("\nBACKWARD PROPOGATION\n")
    for key, val in gradients.items():
        print(key, ": ", val)


if __name__ == "__main__":
    main()
    

    
