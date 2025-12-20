import torch
import numpy as np

def get_random_problems(batch_size, problem_size):
    problems = torch.rand(size=(batch_size, problem_size, 6))
    # problems.shape: (batch, problem, 6)
    return problems

 
