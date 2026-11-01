import pandas as pd

def load_teotil(fname, nutrient_load_varname):
    """
    Load a TEOTIL results file.

    nutrient_load_varname could be e.g. 'accum_all_sources_tot-n_tonnes'

    As of early 2022, these are only useful at the regine unit scale (e.g. 168),
    not in their subunits.
    """
    df = pd.read_csv(fname)
    df = df.loc[df.regine.str.fullmatch('[0-9]{3}\.')]
    df['regine'] = df.regine.str[:3]

    g_N_per_mol = 14.007
    mol_N_per_g = 1/g_N_per_mol
    mmol_N_per_kg = mol_N_per_g * 1e6

    totN_tonnes_per_yr = df[nutrient_load_varname]
    total_river_flux_m3_per_yr = df['accum_q_m3/s'] * 86400 * 365

    conc_kg_per_m3 = 1000 * totN_tonnes_per_yr / total_river_flux_m3_per_yr

    df['conc_nutrient_mmol_per_m3'] = conc_kg_per_m3 * mmol_N_per_kg

    return df[['regine', 'conc_nutrient_mmol_per_m3']]
