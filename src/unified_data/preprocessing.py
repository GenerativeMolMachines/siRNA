import json
import pandas as pd

def preprocess(df: pd.DataFrame):

    def replace_quotes(series):
        def fix_cell(table, col):
            table[col] = table[col].replace("'", '"')
            return series

        series = fix_cell(series, "siRNA sense")
        series = fix_cell(series, "siRNA antisense")
        series = fix_cell(series, "Modification sense")
        series = fix_cell(series, "Modification antisense")
        series = fix_cell(series, "Position sense")
        series = fix_cell(series, "Position antisense")

        return series

    df = df.apply(replace_quotes, axis=1)

    def rewrite_mod_notations(series):

        def get_nucleotides_list(line, col):
            raw_nucleotides = json.loads(line[col])
            nucleotides = [n.split("-")[0] for n in raw_nucleotides]
            json_dump = json.dumps(nucleotides)
            line[col] = json_dump
            return line
        
        def get_mod_names(line, col):
            line[col] = json.dumps(json.loads(line[col]))
            return line
        
        def get_mod_positions(line, col):
            raw_positions = json.loads(line[col])
            int_positions = [list(map(int, pos)) for pos in raw_positions]
            line[col] = int_positions
            return line

        series = get_nucleotides_list(series, "siRNA sense")
        series = get_nucleotides_list(series, "siRNA antisense")
        series = get_mod_names(series, "Modification sense")
        series = get_mod_names(series, "Modification antisense")
        series = get_mod_positions(series, "Position sense")
        series = get_mod_positions(series, "Position antisense")

        return series

    df = df.apply(rewrite_mod_notations, axis=1)

    def get_mod_pos(series):
    
        def mod_pos_for_sequence(line, mod_col, pos_col, result_col):
            
            mods = json.loads(line[mod_col])

            # Strip mod names
            for i, position_mods_list in enumerate(mods):
                for j, m in enumerate(position_mods_list):
                    mods[i][j] = m.strip()

            poses = line[pos_col]
            line[result_col] = [i for i in zip(mods, poses)]
            return line
        
        series = mod_pos_for_sequence(series, mod_col="Modification sense", pos_col="Position sense", result_col="sense mod_pos")
        series = mod_pos_for_sequence(series, mod_col="Modification antisense", pos_col="Position antisense", result_col="antisense mod_pos")

        return series

    df.insert(7, "sense mod_pos", None)
    df.insert(8, "antisense mod_pos", None)
    df = df.apply(get_mod_pos, axis=1)
    df = df.rename(columns={"SMDB_id": "SMDBid"})
    df = df.set_index("SMDBid")
    return df



