import torch
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP:
    def __init__(
        self,
        linear_1_in_features,
        linear_1_out_features,
        f_function,
        linear_2_in_features,
        linear_2_out_features,
        g_function
    ):
        """
        Args:
            linear_1_in_features: the in features of first linear layer
            linear_1_out_features: the out features of first linear layer
            linear_2_in_features: the in features of second linear layer
            linear_2_out_features: the out features of second linear layer
            f_function: string for the f function: relu | sigmoid | identity
            g_function: string for the g function: relu | sigmoid | identity
        """
        self.f_function = f_function
        self.g_function = g_function

        self.parameters = dict(
            W1 = torch.randn(linear_1_out_features, linear_1_in_features),
            b1 = torch.randn(linear_1_out_features),
            W2 = torch.randn(linear_2_out_features, linear_2_in_features),
            b2 = torch.randn(linear_2_out_features),
        )
        self.grads = dict(
            dJdW1 = torch.zeros(linear_1_out_features, linear_1_in_features),
            dJdb1 = torch.zeros(linear_1_out_features),
            dJdW2 = torch.zeros(linear_2_out_features, linear_2_in_features),
            dJdb2 = torch.zeros(linear_2_out_features),
        )

        # put all the cache value you need in self.cache
        self.cache = dict()

        self.activation = dict(
            relu = nn.ReLU(),
            sigmoid = nn.Sigmoid(),
            identity = nn.Identity()
        )

        self.activation_grad = dict(
            relu = lambda x: (x >= 0).float(),
            sigmoid = lambda x: (1/(1 + torch.exp(-x))) * (1 - (1/(1 + torch.exp(-x)))),
            identity = lambda x: torch.ones(x.shape)
        )


    def forward(self, x):
        """
        Args:
            x: tensor shape (batch_size, linear_1_in_features)
        """
        # TODO: Implement the forward function
        self.f = self.activation[self.f_function]
        self.g = self.activation[self.g_function]

        z1 = torch.matmul(x, self.parameters['W1'].t()) + self.parameters['b1'].t()
        z2 = self.f(z1)
        z3 = torch.matmul(z2, self.parameters['W2'].t()) + self.parameters['b2'].t()
        self.y_hat = self.g(z3)

        self.cache['x'] = x
        self.cache['dy_hatdz3'] = self.activation_grad[self.g_function](z3)
        self.cache['dz2dz1'] = self.activation_grad[self.f_function](z1)
        self.cache['z2'] = z2

        return self.y_hat
    
    def backward(self, dJdy_hat):
        """
        Args:
            dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
        """
        # TODO: Implement the backward function
        self.grads['dJdW1'] = torch.matmul(self.cache['x'].t(), torch.matmul(dJdy_hat * self.cache['dy_hatdz3'], self.parameters['W2']) * self.cache['dz2dz1'])
        self.grads['dJdb1'] = torch.sum(torch.matmul(dJdy_hat * self.cache['dy_hatdz3'], self.parameters['W2']) * self.cache['dz2dz1'], dim=0)
        self.grads['dJdW2'] = torch.matmul(self.cache['z2'].t(), dJdy_hat * self.cache['dy_hatdz3'])
        self.grads['dJdb2'] = torch.sum(dJdy_hat * self.cache['dy_hatdz3'], dim=0)
        self.grads['dJdW1'] = self.grads['dJdW1'].t() / (dJdy_hat.shape[0] * dJdy_hat.shape[1])
        self.grads['dJdb1'] = self.grads['dJdb1'].t() / (dJdy_hat.shape[0] * dJdy_hat.shape[1])
        self.grads['dJdW2'] = self.grads['dJdW2'].t() / (dJdy_hat.shape[0] * dJdy_hat.shape[1])
        self.grads['dJdb2'] = self.grads['dJdb2'].t() / (dJdy_hat.shape[0] * dJdy_hat.shape[1])

    
    def clear_grad_and_cache(self):
        for grad in self.grads:
            self.grads[grad].zero_()
        self.cache = dict()

def mse_loss(y, y_hat):
    """
    Args:
        y: the label tensor (batch_size, linear_2_out_features)
        y_hat: the prediction tensor (batch_size, linear_2_out_features)

    Return:
        J: scalar of loss
        dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
    """
    # TODO: Implement the mse loss
    y = y.float()
    J = torch.mean((y_hat - y)**2)
    dJdy_hat = y_hat - y

    return J, dJdy_hat

def bce_loss(y, y_hat):
    """
    Args:
        y_hat: the prediction tensor
        y: the label tensor
        
    Return:
        loss: scalar of loss
        dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
    """
    # TODO: Implement the bce loss
    y = y.float()
    J = torch.mean(-(y * torch.clamp(torch.log(y_hat), min=-100) + (1 - y) * torch.clamp(torch.log(1 - y_hat), min=-100)))
    dJdy_hat = -(y / y_hat) + ((1 - y) / (1 - y_hat))
    
    return J, dJdy_hat
