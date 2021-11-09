import torch
import torch.distributions as dist
from pyrsistent import pmap, PMap

# Scheme type objects
Symbol = str              # A Scheme Symbol is implemented as a Python str
Number = (torch.int32, torch.float64, torch.float32)     # A Scheme Number is implemented as a Python int or float
Atom   = (Symbol, Number) # A Scheme Atom is a Symbol or Number
List   = torch.tensor             # A Scheme List is implemented as a Python list
Expression  = (Atom, List)     # A Scheme expression is an Atom or List 

class Normal(object):
    def __init__(self, alpha, *x):
        self.dist = torch.distributions.Normal(*torch.FloatTensor([*x]))

    def sample(self):
        return self.dist.sample()

    def log_prob(self, c):
        return self.dist.log_prob(c)

class Beta(object):
    def __init__(self, alpha, *x):
        self.dist = torch.distributions.Beta(*torch.FloatTensor([*x]))

    def sample(self):
        return self.dist.sample()

    def log_prob(self, c):
        return self.dist.log_prob(c)

class Exponential(object):
    def __init__(self, alpha, *x):
        self.dist = torch.distributions.Exponential(*torch.FloatTensor([*x]))

    def sample(self):
        return self.dist.sample()

    def log_prob(self, c):
        return self.dist.log_prob(c)

# class UniformContinuous(dist.Gamma):
#     """Gamma approximation of Uniform"""

#     def __init__(self, alpha, lb, ub):
#         if lb == 0:
#             self.dist = torch.distributions.Gamma(concentration=torch.tensor(0.001), rate=ub)
#         else:
#             self.dist = torch.distributions.Gamma(concentration=lb, rate=ub)
    
#     def sample(self):
#         return self.dist.sample()

#     def log_prob(self, x):
#         return self.dist.log_prob(x)

class Uniform(object):
    def __init__(self, alpha, *x):
        self.dist = torch.distributions.Uniform(*torch.FloatTensor([*x]))

    def sample(self):
        return self.dist.sample()

    def log_prob(self, c):
        return self.dist.log_prob(c)


class Gamma(object):
    def __init__(self, alpha, *x):
        self.dist = torch.distributions.Gamma(*torch.FloatTensor([*x]))

    def sample(self):
        return self.dist.sample()

    def log_prob(self, c):
        return self.dist.log_prob(c)

class Dirac(object):
    def __init__(self, alpha, x):
        self.value = x

    def sample(self):
        return self.value

    def log_prob(self, c):
        if c==self.value:
            return torch.tensor(1)
        else:
            return torch.tensor(0)

# class Dirac(object):
#     # approximation to Dirac with Laplace
#     def __init__(self, alpha, x):
#         self.dist = torch.distributions.Laplace(x, torch.tensor([0.25]))

#     def sample(self):
#         return self.dist.sample()

#     def log_prob(self, c):
#         return self.dist.log_prob(c)

class Bernoulli(object):
    def __init__(self, alpha, x):
        self.dist = torch.distributions.Bernoulli(x)
        self.p = x

    def sample(self):
        return self.dist.sample()

    def log_prob(self, c):
        if c:
            return torch.log(self.p)
        else:
            return torch.log(1 - self.p)

class Dirichlet(object):
    def __init__(self, alpha, x):
        self.dist = torch.distributions.Dirichlet(x)

    def sample(self):
        return self.dist.sample()

    def log_prob(self, c):
        return self.dist.log_prob(c)

class Discrete(object):
    def __init__(self, alpha, x):
        self.dist = torch.distributions.Categorical(probs = torch.FloatTensor(x))

    def sample(self):
        return self.dist.sample()

    def log_prob(self, c):
        return self.dist.log_prob(c)

#these are adapted from Peter Norvig's Lispy, from HW6 starter code
class Env():
    "An environment: a dict of {'var': val} pairs, with an outer Env."
    def __init__(self, parms=(), args=(), outer=None):
        self.data = pmap(zip(parms, args))
        self.outer = outer
        if outer is None:
            self.level = 0
        else:
            self.level = outer.level+1

    def __getitem__(self, item):
        return self.data[item]

    def find(self, var):
        "Find the innermost Env where var appears."
        if (var in self.data):
            return self
        else:
            if self.outer is not None:
                return self.outer.find(var)
            else:
                raise RuntimeError('var "{}" not found in outermost scope'.format(var))

    def print_env(self, print_lowest=False):
        print_limit = 1 if print_lowest == False else 0
        outer = self
        while outer is not None:
            if outer.level >= print_limit:
                print('Scope on level ', outer.level)
                if 'f' in outer:
                    print('Found f, ')
                    print(outer['f'].body)
                    print(outer['f'].parms)
                    print(outer['f'].env)
                print(outer,'\n')
            outer = outer.outer

def get(_ , x,i):
    try:
        return x[i.item()]
    except:
        return x[i]

def put(_, x,i,v):
    try:
        x[i.item()] = v
    except:
        x[i] = v
    return x

def append(_, x, v):
    try:
        return torch.stack(list(x) + [v])
    except:
        return torch.stack([x,v])

def prepend(_, x, v):
    try:
        return torch.stack([v] + list(x))
    except:
        return torch.stack([v,x])

def hash_map(_,*x):
    # print("init hashmap", x)
    try:
        return {x[i].item(): x[i + 1] for i in range(0, len(x), 2)}
    except:
        return {x[i]: x[i + 1] for i in range(0, len(x), 2)}


def vector(_, *x):
    try:
        return torch.stack([*x])
    except:
        return x
            
def last(_, x):
    try:
        return x[-1]
    except:
        return x

def first(_, x):
    try:
        return x[0]
    except:
        return x   

def conj(_, x, v):
    try:
        return torch.stack(list(x) + [v])
    except:
        return torch.stack([x,v])

def cons(_, x, v):
    try:
        return torch.stack([v] + list(x) )
    except:
        return torch.stack([v, x])


def push_addr(alpha, value):
    return alpha + value 

def standard_env():
    "An environment with some Scheme standard procedures."
    env = Env(penv.keys(), penv.values())
    return env
        
penv = {
    '>':   lambda a,x,y: torch.gt(x,y), 
    '<':   lambda a,x,y: torch.lt(x,y), 
    '>=':   lambda a,x,y: torch.ge(x,y), 
    '<=':   lambda a,x,y: torch.le(x,y), 
    '=':   lambda a,x,y: torch.eq(x,y), 
    '+':   lambda a,x,y: torch.add(x,y),
    '-':   lambda a,x,y: torch.sub(x,y),
    '*':   lambda a,x,y: torch.mul(x,y),
    '/':   lambda a,x,y: torch.div(x,y),
    'log':   lambda a,x: torch.log(x),
    'sqrt':   lambda a,x: torch.sqrt(x),
    'abs':     lambda a,x: torch.abs(x),
    'apply':   lambda a, proc, args: proc(*args),
    'begin':   lambda a, *x: x[-1],
    'car':     lambda a, x: x[0],
    'cdr':     lambda a, x: x[1:], 
    'second':  lambda a, x: x[1],
    'rest':    lambda a, x: x[1:],
    'list':    lambda a, *x: List(x), 
    'empty?':   lambda a, x: len(x) == 0, 
    '=':        lambda a, x, y: x==y,
    'and':     lambda a, x, y: x and y,
    'or':     lambda a, x, y: x or y,
    'list?':   lambda a, x: isinstance(x, List), 
    'number?': lambda a, x: isinstance(x, Number),  
    'symbol?': lambda a, x: isinstance(x, Symbol),
    'vector':  vector,
    'first':  first,
    'cons':    cons,
    'conj':    conj,
    'last':  last,
    'peek':  first,
    'get':  get,
    'put': put,
    'append': append,
    'hash-map': hash_map,
    'push-address': push_addr,
    # distributions
    # 'uniform-continuous': lambda *x: UniformContinuous(*x),
    'normal': lambda *x: Normal(*x),
    'beta': lambda *x: Beta(*x),
    'uniform': lambda *x: Uniform(*x),
    'uniform-continuous': lambda *x: Uniform(*x),
    'exponential': lambda *x: Exponential(*x),
    'discrete': lambda *x: Discrete(*x),
    'dirichlet': lambda *x: Dirichlet(*x),
    'flip': lambda *x: Bernoulli(*x),
    'gamma': lambda *x: Gamma(*x),
    'dirac': lambda *x: Dirac(*x),
    # matrix operations
    'mat-transpose': lambda x: x.T,
    'mat-tanh': lambda x: x.tanh(),
    'mat-mul': lambda x,y: torch.matmul(x.float(),y.float()),
    'mat-add': lambda x, y: x+y,
    'mat-repmat': lambda x,y,z: x.repeat((int(y), int(z)))
}