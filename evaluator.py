from primitives import env as penv
from daphne import daphne
from tests import is_tol, run_prob_test,load_truth
from pyrsistent import pmap,plist
from primitives import standard_env, Symbol, Env
import torch



global_env = standard_env() 

class Procedure(object):
    "A user-defined Scheme procedure."
    def __init__(self, parms, body, sigma, env):
        self.parms, self.body, self.sigma, self.env = parms, body, sigma, env
    def __call__(self, *args): 
        return eval(self.body, self.sigma, Env(self.parms, args, self.env))


def evaluate_program(ast, sigma = {}):
    """Evaluate a program as desugared by daphne, generate a sample from the prior
    Args:
        ast: json FOPPL program
    Returns: sample from the prior of ast
    """
    for i in range(len(ast)):
        ei, sigma = eval(ast[i],sigma)
        if ei != None:
            res = ei
    return res, sigma

def eval(x, sigma, env=global_env):
    "Evaluate an expression in an environment."
    if isinstance(x, Symbol):    # variable reference
        return env.find(x)[x], sigma
    elif not isinstance(x, list): # constant 
        return torch.tensor(x), sigma 
    op, *args = x       
    if op == 'quote':            # quotation
        return args[0]
    if op == 'if':             # conditional
        (test, conseq, alt) = args
        res, sigma = eval(test, sigma, env)
        exp = (conseq if res else alt)
        return eval(exp, sigma, env)
    elif op == 'define':         # definition
        (symbol, exp) = args
        env[symbol] = eval(exp, env)
    elif op == 'set!':           # assignment
        (symbol, exp) = args
        env.find(symbol)[symbol] = eval(exp, env)
    elif op == 'lambda':         # procedure
        (parms, body) = args
        return Procedure(parms, body, env)
    elif op == 'defn':         # definition
        (string, parms, body) = args
        env[string] = Procedure(parms, body, sigma, env)
        return None, sigma
    elif op == 'sample':
        dist, sigma = eval(args[0], sigma, env)
        return dist.sample(), sigma
    else:                        # procedure call
        proc, sigma = eval(op, sigma, env)
        # TODO: is this the real sigma we want? Idk. 
        #   maybe we should ignore instead of store. Not sure. 
        vals = [x[0] for x in (eval(arg, sigma, env) for arg in args)]
        if type(proc)==Procedure:   # user defined
            r, _ = proc(*vals)
        else:                       # primitive
            r = proc(*vals)
        return r, sigma
    # else:                        # procedure call
    #     proc = eval(op, env)
    #     vals = [eval(arg, env) for arg in args]
    #     return proc(*vals)

def get_stream(exp):
    """Return a stream of prior samples"""
    while True:
        yield evaluate_program(exp)


def run_deterministic_tests():
    
    for i in range(1,14):

        exp = daphne(['desugar-hoppl', '-i', '../../HW5/programs/tests/deterministic/test_{}.daphne'.format(i)])
        truth = load_truth('programs/tests/deterministic/test_{}.truth'.format(i))
        ret = evaluate_program(exp)
        try:
            assert(is_tol(ret, truth))
        except:
            raise AssertionError('return value {} is not equal to truth {} for exp {}'.format(ret,truth,exp))
        
        print('FOPPL Tests passed')
        
    for i in range(1,13):

        exp = daphne(['desugar-hoppl', '-i', '../../HW5/programs/tests/hoppl-deterministic/test_{}.daphne'.format(i)])
        truth = load_truth('programs/tests/hoppl-deterministic/test_{}.truth'.format(i))
        ret = evaluate_program(exp)
        try:
            assert(is_tol(ret, truth))
        except:
            raise AssertionError('return value {} is not equal to truth {} for exp {}'.format(ret,truth,exp))
        
        print('Test passed')
        
    print('All deterministic tests passed')
    


def run_probabilistic_tests():
    
    num_samples=1e4
    max_p_value = 1e-2
    
    for i in range(1,7):
        exp = daphne(['desugar-hoppl', '-i', '../../HW5/programs/tests/probabilistic/test_{}.daphne'.format(i)])
        truth = load_truth('programs/tests/probabilistic/test_{}.truth'.format(i))
        
        stream = get_stream(exp)
        
        p_val = run_prob_test(stream, truth, num_samples)
        
        print('p value', p_val)
        assert(p_val > max_p_value)
    
    print('All probabilistic tests passed')    



if __name__ == '__main__':
    
    run_deterministic_tests()
    # run_probabilistic_tests()
    

    # for i in range(1,4):
    #     print(i)
    #     exp = daphne(['desugar-hoppl', '-i', '../../HW5/programs/{}.daphne'.format(i)])
    #     print('\n\n\nSample of prior of program {}:'.format(i))
    #     print(evaluate_program(exp))        
