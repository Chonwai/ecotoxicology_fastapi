import torch
from service.GCN import GCN, load_model


def serialize_model(model_type, output_path):
    model = load_model(model_type)
    torch.save({
        'state_dict': model.state_dict(),
        'hidden_channels': model.conv1.out_channels,
        'num_node_features': model.conv1.in_channels,
        'num_classes': model.lin.out_features
    }, output_path)
    print(f"Model {model_type} has been serialized to {output_path}.")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python serialize_model.py <model_type> <output_path>")
        sys.exit(1)
    model_type = sys.argv[1]
    output_path = sys.argv[2]
    serialize_model(model_type, output_path)
