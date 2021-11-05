import torch
import torch.distributions as dist
from pyrsistent import pmap,plist
import torch
import operator as op

# rest, nth, conj, cons as described in the book

# Scheme type objects
Symbol = str              # A Scheme Symbol is implemented as a Python str
Number = (torch.int32, torch.float64, torch.float32)     # A Scheme Number is implemented as a Python int or float
Atom   = (Symbol, Number) # A Scheme Atom is a Symbol or Number
List   = torch.tensor             # A Scheme List is implemented as a Python list
Expression  = (Atom, List)     # A Scheme expression is an Atom or List

class Env(dict):
    "An environment: a dict of {'var': val} pairs, with an outer Env."
    def __init__(self, parms=(), args=(), outer=None):
        self.update(zip(parms, args))
        self.outer = outer
    def find(self, var):
        "Find the innermost Env where var appears."
        try: 
            return self if (var in self) else self.outer.find(var)
        except: 
            print("except", var)
            raise
        
def get(x,i):
    if type(x)==dict:
        return x[i.item()]
    else:
        return x[int(i)]

def put(x,i,v):
    if type(x)==dict:
        x[i.item()] = v
    else:
        x[int(i)] = v
    return x

def append(x, v):
    try:
        return torch.stack(list(x) + [v])
    except:
        return torch.stack([x,v])

def prepend(x, v):
    try:
        return torch.stack([v] + list(x))
    except:
        return torch.stack([v,x])

def hash_map(*x):
    return {x[i].item(): x[i + 1] for i in range(0, len(x), 2)}

def hash_map_graph(x):
    return {x[i].item(): x[i + 1] for i in range(0, len(x), 2)}

def vector(*x):
    try:
        return torch.stack([*x])
    except:
        return x
            
def last(x):
    try:
        return x[-1]
    except:
        return x

def first(x):
    try:
        return x[0]
    except:
        return x

def standard_env():
    "An environment with some Scheme standard procedures."
    env = pmap(standard_env_init())
    return env
    
def standard_env_init() -> Env:
    "An environment with some Scheme standard procedures."
    env = Env()
    env.update({'alpha' : ''}) 
    env.update(vars(torch)) # sin, cos, sqrt, pi, ...
    env.update({
        '+':torch.add, '-':torch.sub, '*':torch.mul, '/':torch.div, 
        '>':torch.gt, '<':torch.lt, '>=':torch.ge, '<=':torch.le, '=':torch.eq, 
        'abs':     abs,
        'apply':   lambda proc, args: proc(*args),
        'begin':   lambda *x: x[-1],
        'car':     lambda x: x[0],
        'cdr':     lambda x: x[1:], 
        'cons':    lambda x,y: [x] + y,
        'eq?':     op.is_, 
        'expt':    pow,
        'equal?':  torch.eq, 
        'length':  len, 
        'list':    lambda *x: List(x), 
        'list?':   lambda x: isinstance(x, List), 
        'map':     map,
        'max':     max,
        'min':     min,
        'not':     op.not_,
        'empty?':   lambda x: x == [], 
        '=':        lambda x, y: x==y,
        'and':     lambda x, y: x and y,
        'or':     lambda x, y: x or y,
        'number?': lambda x: isinstance(x, Number),  
		'print':   print,
        'procedure?': callable,
        'round':   round,
        'symbol?': lambda x: isinstance(x, Symbol),
        'vector':  vector,
        'first':  first,
        'last':  last,
        'second': lambda x: x[1],
        'rest':  lambda x: x[1:],
        'get':  get,
        'put': put,
        'append': append,
        'hash-map': hash_map,
        'hash-map-graph': hash_map_graph,
        'normal': lambda *x: Normal(*x),
        # 'beta': lambda *x: Beta(*x),
        # 'exponential': lambda *x: Exponential(*x),
        # 'uniform': lambda *x: Uniform(*x),
        # 'discrete': lambda *x: Discrete(*x),
        # 'dirichlet': lambda *x: Dirichlet(*x),
        # 'flip': lambda *x: Bernoulli(*x),
        # 'gamma': lambda *x: Gamma(*x),
        # 'dirac': lambda *x: Dirac(*x),
        'mat-transpose': lambda x: x.T,
        'mat-tanh': lambda x: x.tanh(),
        'mat-mul': lambda x,y: torch.matmul(x.float(),y.float()),
        'mat-add': lambda x, y: x+y,
        'mat-repmat': lambda x,y,z: x.repeat((int(y), int(z)))
    })
    return env


class Normal(dist.Normal):
    
    def __init__(self, alpha, loc, scale):
        
        if scale > 20.:
            self.optim_scale = scale.clone().detach().requires_grad_()
        else:
            self.optim_scale = torch.log(torch.exp(scale) - 1).clone().detach().requires_grad_()
        
        
        super().__init__(loc, torch.nn.functional.softplus(self.optim_scale))
    
    def Parameters(self):
        """Return a list of parameters for the distribution"""
        return [self.loc, self.optim_scale]
        
    def make_copy_with_grads(self):
        """
        Return a copy  of the distribution, with parameters that require_grad
        """
        
        ps = [p.clone().detach().requires_grad_() for p in self.Parameters()]
         
        return Normal(*ps)
    
    def log_prob(self, x):
        
        self.scale = torch.nn.functional.softplus(self.optim_scale)
        
        return super().log_prob(x)
        

def push_addr(alpha, value):
    return alpha + value



env = {
           'normal' : Normal,
           'push-address' : push_addr,
       }






