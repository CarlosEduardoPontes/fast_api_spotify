import pandas as pd

def dict_to_pandas(payload, list_fields):
    df = pd.DataFrame(payload, index=[0])
    return df[list_fields]
