import torch
import torch.nn as nn
def main():
    # I want a linear layer of shape 2, 2, with weights 1, 2, 3 and 4    
    linear = nn.Linear(2, 2)
    linear.weight.data = torch.tensor([[1, 2], [3, 4]], dtype=torch.float)
    linear.bias.data = torch.tensor([0, 0], dtype=torch.float)
    print(linear.weight)

    test_input = torch.tensor([[1, 2]], dtype=torch.float)
    
    ouput = linear(test_input)
    
    print(test_input @ linear.weight.data.t())
    print(ouput)
    
    


    
main()