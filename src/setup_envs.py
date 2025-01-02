import os

SRC_PATH = os.getcwd()
DATA_PATH = os.path.join(SRC_PATH, "datasets")

envs = (
    f"{SRC_PATH=}\n"
    f"{DATA_PATH=}\n"
)

if __name__ == "__main__":
    with open("./.env", "w") as f:
        f.write(envs)
