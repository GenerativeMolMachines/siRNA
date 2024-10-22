import json

import pandas as pd
import numpy as np

from rdkit import Chem
from CDK_pywrapper import CDK as CDK_wrap


def get_uniqie_smiles(df: pd.DataFrame) -> list[str]:

    sense = [json.loads(i) for i in df.Sense]
    antisense = [json.loads(i) for i in df.AntiSense]
    unique_smiles = []
    for seq in sense + antisense:
        for nuc in seq:
            if nuc not in unique_smiles:
                unique_smiles.append(nuc)

    return unique_smiles


class CDK:

    unique_smiles: dict[str, np.ndarray] = None

    @classmethod
    def smiles2descriptors(cls, smiles: str) -> np.ndarray:

        mol = [Chem.AddHs(Chem.MolFromSmiles(smiles))]
        cdk = CDK_wrap()

        try:
            descriptors = cdk.calculate(mol).loc[0].values
        except Exception as ex:
            print("\033[93m", f"Unable to generate descriptors for SMILES: {smiles}", "\033[0m")
            return

        descriptors = np.array(descriptors)
        num_descriptors = descriptors.shape[0]
        descriptors = descriptors.reshape((-1, num_descriptors))
        return descriptors
    
    @classmethod
    def unique_smiles2descriptors(cls, unique_smiles: list[str]) -> tuple[np.ndarray, bool]:

        unique_smiles_desc = {}
        for smiles in unique_smiles:
            desc = cls.smiles2descriptors(smiles)
            if desc is not None:
                unique_smiles_desc[smiles] = desc
        cls.unique_smiles = unique_smiles_desc
        return unique_smiles_desc
    
    @classmethod
    def load_unique(cls, unique_smiles: list[str]):

        filename = "data/unique_cdk_descriptors.csv"
        df = pd.read_csv(filename, index_col=0, header=0)

        cls.unique_smiles = {}
        for _, line in df.iterrows():
            cls.unique_smiles[line.name] = line.to_numpy()

        num_descriptors = list(cls.unique_smiles.values())[0].shape[0]
        for smiles in cls.unique_smiles:
            cls.unique_smiles[smiles] = cls.unique_smiles[smiles].reshape((-1, num_descriptors))
    
    
    @classmethod
    def encode_sequence(cls, sequence: list[str], max_length: int = 27):

        num_descriptors = list(cls.unique_smiles.values())[0].shape[1]
        
        sequence_desc = np.empty((0, num_descriptors))
        invalid = False
        for smiles in sequence:
            if smiles not in cls.unique_smiles.keys():
                invalid = True
                break
            desc = cls.unique_smiles[smiles]
            sequence_desc = np.vstack((sequence_desc, desc))

        if invalid is True:
            sequence_desc = np.zeros((num_descriptors * max_length))
        else:
            for _ in range(max_length - len(sequence)):
                sequence_desc = np.vstack((sequence_desc, np.zeros((num_descriptors))))
            sequence_desc = sequence_desc.flatten()
        
        return sequence_desc, invalid
    
    @classmethod
    def get_dataset(cls, df: pd.DataFrame, max_length: int = 27) -> tuple[np.ndarray]:
        
        unique_smiles = get_uniqie_smiles(df)
        # cls.unique_smiles2descriptors(unique_smiles)
        cls.load_unique(unique_smiles)

        num_descriptors = list(cls.unique_smiles.values())[0].shape[1]

        x = np.empty((0, num_descriptors * max_length * 2))
        experiment_data = np.empty((0, 6))
        y = np.empty((0, 1))

        invalid_cases = []
        def process_invalid_case(line: pd.Series) -> None:
            nonlocal x, experiment_data, y, invalid_cases
            x = np.vstack((x, np.zeros(num_descriptors * max_length * 2)))
            y = np.vstack((y, np.array([None])))
            experiment_data = np.vstack((experiment_data, np.zeros(6)))
            invalid_cases.append(line.name)
        
        for _, line in df.iterrows():

            sense = json.loads(line.Sense)
            x_sense, invalid = cls.encode_sequence(sense, max_length=max_length)

            if invalid is True:
                process_invalid_case(line)
                continue
            else:
                antisense = json.loads(line.AntiSense)
                x_antisense, invalid = cls.encode_sequence(antisense, max_length=max_length)
                if invalid is True:
                    process_invalid_case(line)
                    continue

            x = np.vstack((x, np.hstack((x_sense, x_antisense))))
            y = np.vstack((y, line["Efficacy, %"]))
            experiment_data = np.vstack((
                experiment_data, 
                np.array([
                    line["siRNA concentration"], 
                    line["Duration after transfection"], 
                    line["Experiment used to check activity"], 
                    line["Target gene"], 
                    line["Cell or Organism used"], 
                    line["Transfection method"],
                ])
            ))

        # clear zero lines (invalid cases)
        x = x[~np.all(x == 0, axis=1)]
        y = y[~np.all(y == None, axis=1)]
        experiment_data = experiment_data[~np.all(experiment_data == 0, axis=1)]

        print(f"Initial dataset: {df.shape}", end="\n\n")
        print(f"Cases skipped: {len(invalid_cases)}")
        print(f"Invalid smdb_ids: {invalid_cases}", end="\n\n")

        print(f"x.shape: {x.shape}")
        print(f"experiment_data.shape: {experiment_data.shape}")
        print(f"y.shape: {y.shape}")

        assert x.shape[0] == y.shape[0] == experiment_data.shape[0]

        return x, experiment_data, y
