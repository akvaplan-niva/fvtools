"""
Helper functions to deal with Regine units
"""

def non_kystfelt_drains_into_sea(regine_id):
    """
    We can determine whether a given non-coastal Regine area drains directly into the sea from its suffix.

    Parameters
    ----------
    regine_id: str
    """
    suffix = regine_id.split('.')[1]
    return fnmatch(suffix, '?Z') or fnmatch(suffix, '?A')

def classify_nedborfelt(df, vassdragnr):
    """
    Classifies a pandas series of regine watershed units into kystfelt and nedborfelt

    Note
    ----
    vassdragnr and df must have the same index

    See https://www.nve.no/media/2297/regines-inndelingssystem.pdf
    """
    df['nedborfelt'] = vassdragnr.str.match('[0-9]{3}.[a-zA-Z][0-9a-zA-Z]*')
    df['kystfelt'] = vassdragnr.str.match('[0-9]{3}.[0-9][0-9A-Z]*')
