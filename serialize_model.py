from service.GCN import GCN, load_model
import pickle


def serialize_model(model_type, output_path):
    model = load_model(model_type)
    with open(output_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model {model_type} has been serialized to {output_path}.")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python serialize_model.py <model_type> <output_path>")
        sys.exit(1)
    model_type = sys.argv[1]
    output_path = sys.argv[2]
    serialize_model(model_type, output_path)
