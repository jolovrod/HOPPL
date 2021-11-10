from daphne import daphne
from tests import is_tol, run_prob_test,load_truth
from primitives import standard_env, Symbol, Env
from plots import plots
import torch
import copy
import sys
import threading

def evaluate_program(ast):
    """Evaluate a program as desugared by daphne, generate a sample from the prior
    Args:
        ast: json FOPPL program
    Returns: sample from the prior of ast
    """
    sigma = {}
    env = standard_env()
    fn = eval(ast, sigma, env)[0]
    return fn("")[0]

class Procedure(object):
    "A user-defined Scheme procedure."
    def __init__(self, parms, body, sigma, env):
        self.parms, self.body, self.sigma, self.env = parms, body, sigma, env
    def __call__(self, *args): 
        newenv = copy.deepcopy(self.env)
        return eval(self.body, self.sigma, Env(self.parms, args, newenv))

def eval(x, sigma, env=standard_env()):
    "Evaluate an expression in an environment."
    if isinstance(x, Symbol):    # variable reference
        if x[0] == x[-1] == '"':
            return x, sigma
        return env.find(x)[x], sigma
    elif not isinstance(x, list): # constant 
        return torch.tensor(x), sigma
    op, params, *args = x  
    if op == 'if':             # conditional
        test = params
        (conseq, alt) = args
        res, sigma = eval(test, sigma, env)
        exp = (conseq if res else alt)
        return eval(exp, sigma, env)
    elif op == 'defn':         # definition
        (string, parms, body) = args
        env[string] = Procedure(parms, body, sigma, env)
        return None, sigma
    elif op == 'push-address':
        return params + args[0], sigma 
    elif op == 'fn':         # local function
        body = args[0]
        return Procedure(params, body, sigma, env), sigma
    elif op == 'sample':
        dist, sigma = eval(args[0], sigma, env)
        return dist.sample(), sigma
    elif op == 'observe':
        dist, sigma = eval(args[0], sigma, env)
        return dist.sample(), sigma
    else:                        # procedure call
        proc, sigma = eval(op, sigma, env)
        vals = [""]
        for x in (eval(arg, sigma, env) for arg in args):
            vals.append(x[0])
        if type(proc)==Procedure:   # user defined
            r, _ = proc(*vals)
        else:                       # primitive
            r = proc(*vals)
        return r, sigma
        # TODO: in inference, is this the correct sigma we want? 
        #       add correct addressing, but irrelevant for sampling from prior 

def get_stream(exp):
    """Return a stream of prior samples"""
    while True:
        yield evaluate_program(exp)


def run_deterministic_tests():
    
    for i in range(1,14):

        exp = daphne(['desugar-hoppl', '-i', 'C:/Users/jlovr/CS532-HW5/HOPPL/programs/tests/deterministic/test_{}.daphne'.format(i)])
        truth = load_truth('C:/Users/jlovr/CS532-HW5/HOPPL/programs/tests/deterministic/test_{}.truth'.format(i))
        ret = evaluate_program(exp)
        try:
            assert(is_tol(ret, truth))
        except:
            raise AssertionError('return value {} is not equal to truth {} for exp {}'.format(ret,truth,exp))

        print('Deterministic test',i, 'passed')
        print("Return:", ret)
        print("\n\n\n")
        
    print('FOPPL Tests passed')
        
    for i in range(1,13):

        exp = daphne(['desugar-hoppl', '-i', 'C:/Users/jlovr/CS532-HW5/HOPPL/programs/tests/hoppl-deterministic/test_{}.daphne'.format(i)])
        truth = load_truth('C:/Users/jlovr/CS532-HW5/HOPPL/programs/tests/hoppl-deterministic/test_{}.truth'.format(i))
        ret = evaluate_program(exp)
        try:
            assert(is_tol(ret, truth))
        except:
            raise AssertionError('return value {} is not equal to truth {} for exp {}'.format(ret,truth,exp))
        
        print('Hoppl-deterministic test',i, 'passed')
        print("return:", ret)
        print("\n\n\n")
        
    print('All deterministic tests passed')


def run_probabilistic_tests():
    
    num_samples=1e4
    max_p_value = 1e-2
    
    for i in range(1,7):
        exp = daphne(['desugar-hoppl', '-i', 'C:/Users/jlovr/CS532-HW5/HOPPL/programs/tests/probabilistic/test_{}.daphne'.format(i)])
        truth = load_truth('C:/Users/jlovr/CS532-HW5/HOPPL/programs/tests/probabilistic/test_{}.truth'.format(i))
        
        stream = get_stream(exp)
        
        p_val = run_prob_test(stream, truth, num_samples)
        
        assert(p_val > max_p_value)
        print('Probabilistic test', i, 'passed')
        print("p value:", p_val)
        print("\n\n\n")

    
    print('All probabilistic tests passed')    

def my_main():
    run_deterministic_tests()
    run_probabilistic_tests()

    n = 200
    
    for i in range(1,4):
        exp = daphne(['desugar-hoppl', '-i', 'C:/Users/jlovr/CS532-HW5/HOPPL/programs/{}.daphne'.format(i)])
        samples = []
        for _ in range(n):
            samples.append(evaluate_program(exp))

        plots(i, samples)

if __name__ == '__main__':
    sys.setrecursionlimit(100000)
    threading.stack_size(200000000)
    thread = threading.Thread(target=my_main)
    thread.start()     