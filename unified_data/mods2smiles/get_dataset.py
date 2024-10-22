import json

import pandas as pd

from .gather import modify_nucleotide


def run_dataset(df: pd.DataFrame, modif_correct: int) -> pd.DataFrame:

    new_df = pd.DataFrame(columns=['SMDBid', 'Sense', 'Antisense'])

    success_counter = 0
    for i, line in df.iterrows():

        print(i, end=" ")
        match modif_correct:
            case 4:
                sense_tuple = (json.loads(line["siRNA sense"]), line["sense mod_pos"])
                antisense_tuple = (json.loads(line["siRNA antisense"]), line["antisense mod_pos"])
            case 3:
                sense_tuple, antisense_tuple = parse_line(line)
            case _:
                raise ValueError(f"unknown modif_correct: {modif_correct}")

        sense_smiles = modify_sequence(*sense_tuple)
        if sense_smiles is not None:
            antisense_smiles = modify_sequence(*antisense_tuple)
            if antisense_smiles is not None:

                sense_json = json.dumps(sense_smiles)
                antisense_json = json.dumps(antisense_smiles)

                new_df.loc[new_df.shape[0]] = [i, sense_json, antisense_json]

                print("\033[92m" + "OK" + "\033[0m")
                success_counter += 1

    # new_df.to_csv(output_filename, index=False)

    print(f"{success_counter = }")
    return new_df


def assrted_load(series, col, string_to_load=None):
    error_alias = {
        "siRNA sense": "sense seq",
        "siRNA antisense": "antisense seq",
        "Modification sense": "mod sense",
        "Modification antisense": "mod antisense",
        "Position sense": "pos sense",
        "Position antisense": "pos antisense",
    }

    try:
        if string_to_load is None:
            loaded = json.loads(series[col])
        else:
            print(f"Trying to load from fixed string: {string_to_load}")
            loaded = json.loads(string_to_load)
        return loaded
    except Exception as ex:
        print(f"Exception: {ex}")
        print(f"{col}: {series[col]}")
        print(f"Code: {error_alias.get(col, 'unkonwn')}")

        # Trying to fix the problem
        original_string = series[col]
        fixed_string = fix_quotes(original_string)
        if original_string != fixed_string:
            loaded = assrted_load(series, col, string_to_load=fixed_string)
            return loaded
        
        raise Exception("Asserted load failed")
    

def fix_quotes(string):
    quotes_ids_to_delete = []
    for i, letter in enumerate(string):
        if letter == '"':
            if i != 0 and string[i-1] == "[":
                continue
            if i != len(string) -1 and string[i+1] == "]":
                continue
            quotes_ids_to_delete.append(i)

    fixed_string = ""
    for i, letter in enumerate(string):
        if i not in quotes_ids_to_delete:
            fixed_string += letter

    print(f"fixed string: {fixed_string}")
    return fixed_string



def parse_line(series):
    try:
        sense = assrted_load(series, "siRNA sense")
        antisense = assrted_load(series, "siRNA antisense")

        mod_sense = assrted_load(series, "Modification sense")
        mod_antisense = assrted_load(series, "Modification antisense")

        pos_sense = assrted_load(series, "Position sense")
        pos_sense = [list(map(int, i)) for i in pos_sense]

        pos_antisense = assrted_load(series, "Position antisense")
        pos_antisense = [list(map(int, i)) for i in pos_antisense]

        mod_pos_sense = [i for i in zip(mod_sense, pos_sense)]
        mod_pos_antisense = [i for i in zip(mod_antisense, pos_antisense)]

    except Exception as ex:
        raise Exception(f"parse_line exception: {ex}")

    return (
        (sense, mod_pos_sense),
        (antisense, mod_pos_antisense)
    )


def modify_sequence(sequence, mod_pos) -> list[str] | None:

    # print(f"sequence: {sequence}")
    # print(f"mod_pos: {mod_pos}", end='\n\n')

    sequence_smiles = []
    for index, nucleotide in enumerate(sequence, 1):
        modifications_on_position = []
        for mod, pos in mod_pos:
            if index in pos:
                modifications_on_position.append(mod)

        modifications_on_position = [i[0] for i in modifications_on_position]
        try:
            modified_nucleotide = modify_nucleotide(nucleotide, modifications_on_position)
        except:
            modified_nucleotide = None
        
        if modified_nucleotide is None:
            return
        sequence_smiles.append(modified_nucleotide)

    return sequence_smiles


if __name__ == "__main__":
    
    # df = pd.read_csv("mod/modif_correct2modified_smiles/modif_correct3.csv", delimiter=";", index_col=1)
    # run_dataset(df, 3)

    df = pd.read_csv("mod/modif_correct2modified_smiles/modif_correct_4.csv", delimiter=",", index_col=0)
    run_dataset(df, 4)
