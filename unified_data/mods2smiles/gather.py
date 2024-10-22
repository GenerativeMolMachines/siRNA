import json


default_phosphate = 'P(=O)(O)OP(=O)(O)OP(=O)(O)O'
default_sugar = 'C3OC(CO{PHOSPHATE})C(O)C(O)3'


def modify_nucleotide(nucleotide: str, modifications: list[str] = None) -> str | None:

    """
    Converts nucleotide and list of modifications to SMILES string

    :param nucleotide:      One of these: 'A', 'C', 'G', 'T', 'U'
    :param modifications:   Example:      ['2-Methoxy', 'inverted abasic']
    :return:                Smiles of modified nucleotide
    """

    modifications = list(map(get_modification_from_json, modifications))

    if not all(modifications):
        print("\033[91m" + "modification not found" + "\033[0m")
        return

    modifications = double_mod_check(modifications)
    whole_mod = whole_mod_check(modifications)

    if whole_mod is not None:
        return get_whole_modification(whole_mod)

    base = default_base(nucleotide)
    if base is None:
        return

    sugar = default_sugar
    phosphate = default_phosphate

    for mod in modifications:

        match mod["place"]:
            case "sugar":
                sugar = modify_sugar(mod)
            case "phosphate":
                phosphate = modify_phosphate(mod)
            case "base":
                base = modify_base(mod)

    modified_nucleotide = combine(base, sugar, phosphate)

    return modified_nucleotide


def get_modification_from_json(mod: str) -> dict | None:
    """
    Loads mods.json file and finds there the modification

    :param mod:     example: '2-Methoxy' or 'inverted abasic' etc
    :return:        dict of modification found in mods.json or None if not found
    """

    file_path = "mods2smiles/mods.json"
    with open(file_path, "r", encoding="utf-8") as file:
        mods = json.load(file)

    for m in mods:
        if mod in m["names"]:
            if m["smiles"]:
                return m
            break


def default_base(nucleotide: str) -> str:

    """
    Converts letter of nucleotide to SMILES string

    :param nucleotide:   One of these: 'A', 'C', 'G', 'T', 'U'
    :return:             Smiles of unmodified nucleotide
    """

    match nucleotide:
        case "A":
            return 'Nc1ncnc2N({RIBOSE})cnc12'
        case "C":
            return 'Nc1ccN({RIBOSE})c(=O)n1'
        case "G":
            return 'Nc1nc2N({RIBOSE})cnc2c(=O)[nH]1'
        case "T":
            return 'CC1=CN(C(=O)NC1=O){RIBOSE}'
        case "U":
            return 'O=c1ccN({RIBOSE})c(=O)N1'
        case _:
            return None


def modify_sugar(mod: dict) -> str:

    """
    Based on locant and smiles of modification (in dict "mod") modifies "default_sugar"

    :param mod:     dict of modification found in mods.json
    :return:        smiles of modified ribose, like "C3OC(CO{PHOSPHATE})C(O)C(OC)3" for 2-Methoxy
    """

    pos3 = "C(O)"
    pos2 = "C(O)"

    match mod["locant"]:
        case "2":
            pos2 = mod["smiles"]
        case "3":
            pos3 = mod["smiles"]
        case "whole":
            return mod["smiles"]

    ribose = f'C3OC(CO[PHOSPHATE]){pos3}{pos2}3'
    ribose = ribose.replace("[PHOSPHATE]", "{PHOSPHATE}")

    return ribose


def modify_phosphate(mod: dict) -> str:
    return mod["smiles"]


def modify_base(mod: dict) -> str:
    return mod["smiles"]


def get_whole_modification(mod):
    return mod["smiles"].replace("{PHOSPHATE}", default_phosphate)


def combine(base: str, sugar: str, phosphate: str) -> str:

    """
    Combines parts of molecule base + sugar + phosphate and returns final smiles

    :param base:        smiles string of modified or unmodified base
    :param sugar:       smiles string of modified or unmodified ribose
    :param phosphate:   smiles string of modified or unmodified phosphate
    :return:            smiles string of final nucleotide
    """

    """Rough pseudocode implementation: """

    return base.replace("{RIBOSE}", sugar.replace('{PHOSPHATE}', phosphate))


def double_mod_check(mods):

    base_mods = [m for m in mods if m["place"] == "base"]
    sugar_mods = [m for m in mods if m["place"] == "sugar"]
    phosphate_mods = [m for m in mods if m["place"] == "phosphate"]
    whole_mods = [m for m in mods if m["place"] == "whole"]

    # print(len(base_mods))
    # print(len(sugar_mods))
    # print(len(phosphate_mods))
    # print(len(whole_mods), end='\n\n')

    new_mods = []
    if len(base_mods) > 1:
        ...
    else:
        new_mods += base_mods

    if len(sugar_mods) > 1:
        ...
    else:
        new_mods += sugar_mods

    if len(phosphate_mods) > 1:
        ...
    else:
        new_mods += phosphate_mods

    if len(whole_mods) > 1:
        ...
    else:
        new_mods += whole_mods

    assert [m["place"] for m in new_mods].count("sugar") <= 1
    assert [m["place"] for m in new_mods].count("base") <= 1
    assert [m["place"] for m in new_mods].count("phosphate") <= 1
    assert [m["place"] for m in new_mods].count("whole") <= 1

    return mods


def whole_mod_check(mods):

    places = [m["place"] for m in mods]
    names = {name for m in mods for name in m["names"]}
    if "whole" in places and len(mods) > 1:

        file_path = "mods2smiles/mod_pair.json"
        with open(file_path, "r", encoding="utf-8") as file:
            mod_pairs = json.load(file)

        pair_found = None
        for mp in mod_pairs:

            if set(mp["pair"]).issubset(names):
                pair_found = mp
                break

        if pair_found is None:
            print("\033[91m" + "Pair modification not found" + "\033[0m")
            raise Exception("ABOBA ABOBA ABOBA")
        else:
            return pair_found

    elif "whole" in places and len(mods) == 1:
        return mods[0]



if __name__ == "__main__":
    x = modify_nucleotide("A", ['Thymidine'])
    print(x)

