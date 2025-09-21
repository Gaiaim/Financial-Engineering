import numpy as np
from datetime import date
from yearfrac import yearfrac, mod

def interpolation(start_date: date, end_date: date, start_B: float, end_B: float, 
                  target_date: date, fwd: float, today: date) -> float:
    """
    Computes the interpolated discount factor for a given target_date using the
    provided yearfrac function with the ACT/365 convention.
    
    Parameters:
    - start_date: The earlier date in the interpolation range.
    - end_date: The later date in the interpolation range.
    - start_B: Discount factor for start_date.
    - end_B: Discount factor for end_date.
    - target_date: The date for which the discount factor is computed.
    - fwd: Forward adjustment factor (usually 1, unless a specific adjustment is needed).
    - today: The valuation date.
    
    Returns:
    - discount: The interpolated discount factor for the target_date.
    """
    # Calcola le frazioni d'anno usando la funzione yearfrac e la convenzione ACT/365
    y_frac_start = yearfrac(today, start_date, mod.ACT_365)
    y_frac_end   = yearfrac(today, end_date, mod.ACT_365)
    y_frac_target = yearfrac(today, target_date, mod.ACT_365)

    # Calcola i tassi zero per i due punti
    eps_start = -np.log(start_B) / y_frac_start
    eps_end   = -np.log(end_B) / y_frac_end

    # Se target_date è compresa tra start_date ed end_date, esegue l'interpolazione lineare
    if start_date < target_date < end_date:
        # Calcola la frazione del periodo trascorso utilizzando yearfrac
        t = yearfrac(start_date, target_date, mod.ACT_365) / yearfrac(start_date, end_date, mod.ACT_365)
        y = eps_start + t * (eps_end - eps_start)
    else:
        # Al di fuori dell'intervallo, si usa il tasso finale (estrapolazione piatta)
        y = eps_end

    # Calcola il fattore di sconto interpolato e applica l'aggiustamento forward
    discount = np.exp(-y * y_frac_target) * fwd
    return discount
