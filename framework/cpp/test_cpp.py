import backend

x = backend.Tensor([[1,2,3],
                    [4,5,6],
                    [7,8,9]])

W = backend.Tensor([[1,0],
                    [0,1]])

b = backend.Tensor(0.0)

y = backend.conv2d(x,W,b)
y.backward()

print("y =", y.data_mat)
print("dx =", x.grad_mat)
print("dW =", W.grad_mat)
