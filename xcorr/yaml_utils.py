import yaml

def read_configuration(filename: str) -> dict:
    with open(filename, 'r') as f:
        data = yaml.safe_load(f)
    return data

#https://stackoverflow.com/a/61411523
def dict_to_namedtuple(typename, data):
    return collections.namedtuple(typename, data.keys())(
        *(dict_to_namedtuple(typename + '_' + k, v) if isinstance(v, dict) else v for k, v in data.items())
    )