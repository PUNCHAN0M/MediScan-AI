import argparse
import os

def create_structure(main_class, subclass):
    # Define the base paths
    base_path = os.path.join("data", main_class, subclass)
    folders = [
        os.path.join(base_path, "train", "good"),
        os.path.join(base_path, "test", "good")
    ]

    for folder in folders:
        try:
            os.makedirs(folder, exist_ok=True)
            print(f"Successfully created: {folder}")
        except Exception as e:
            print(f"Error creating {folder}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a dataset folder structure.")
    parser.add_argument("--main_class", required=True, help="The main category name")
    parser.add_argument("--subclass", required=True, help="The subcategory name")

    args = parser.parse_args()
    create_structure(args.main_class, args.subclass)
    # python structure.py --main_class electronics --subclass phones