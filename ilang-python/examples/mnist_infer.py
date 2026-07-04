import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from ilang import i
from mnist import MLP


BATCH_SIZE = 8

# components (o: out_features, i: in_features, b: batch)
perm = i.I | ~(i.I | i.I) # permute the inputs from w,x,b to w,b,x
linear = perm >> ((i("oi*bi~boi") >> i("+boi~bo")) | i.I) >> i("bo+o~bo")
relu = i(">bo~bo")

def i_mlp(x, state):
    # bind weights into linear components
    fc1 = linear(state["fc1.weight"], state["fc1.bias"])
    fc2 = linear(state["fc2.weight"], state["fc2.bias"])
    fc3 = linear(state["fc3.weight"], state["fc3.bias"])

    x = x.view(x.size(0), 28 * 28).contiguous()

    # Torch-like "reassignment" style also works
    # x = relu(fc1(x))
    # x = relu(fc2(x))
    # x = fc3(x)
    # return x()

    # i combinator style
    mlp = fc1 >> relu >> fc2 >> relu >> fc3
    return mlp(x)()

def main():
    state = torch.load("mnist_mlp.pt", map_location="cpu")

    test_data = datasets.MNIST("data", train=False, download=True, transform=transforms.ToTensor())
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    x, y = next(iter(test_loader))

    # Run inference with the 𝚒 implementation.
    with torch.no_grad():
        i_logits = i_mlp(x, state)
        i_pred = i_logits.argmax(dim=1)

    print("labels:       ", y.tolist())
    print("i predictions:", i_pred.tolist())
    print("i logits shape:", tuple(i_logits.shape))

    # Optional sanity check against the original PyTorch model.
    model = MLP()
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        torch_logits = model(x)
        torch_pred = torch_logits.argmax(dim=1)

    print("torch preds:  ", torch_pred.tolist())
    print("max logit diff:", (i_logits - torch_logits).abs().max().item())


if __name__ == "__main__":
    main()
