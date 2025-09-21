import numpy as np
import pandas as pd
from datetime import date, datetime
from scipy.interpolate import CubicSpline

# Importa le funzioni già definite nei rispettivi file
from yearfrac import yearfrac, mod            # mod contiene ACT_360, ACT_365, EU_30_360
from interpolation import interpolation
from add_Dates import add_Dates, mod as mod_adjust  # mod_adjust per l'aggiustamento delle date business

def to_date(x):
    """
    Converte un oggetto in datetime.date se è un pd.Timestamp o datetime.datetime.
    Se l'oggetto è già un datetime.date, lo restituisce così com'è.
    """
    if isinstance(x, pd.Timestamp):
        return x.date()
    if isinstance(x, datetime):
        return x.date()
    return x

def bootstrap(datesSet, ratesSet):
    """
    Esegue il bootstrapping per calcolare i discount factors a partire dai tassi di Depos, Futures e Swaps.
    
    Si assume che:
      - datesSet abbia gli attributi:
          • settle: data di valutazione (oggetto date o Timestamp)
          • depos: DataFrame con la colonna "Settle Dates" (Depos)
          • future: DataFrame con le colonne "Settle" ed "Expiry" (Futures)
          • swap: DataFrame con la colonna "Swap Dates" (Swaps)
      - ratesSet abbia gli attributi:
          • depos: DataFrame con colonne "Bid", "Ask", "Mid" (Depos)
          • future: DataFrame con colonne "Bid", "Ask", "Mid" (Futures)
          • swap: DataFrame con colonne "Bid", "Ask", "Mid" (Swaps)

    Nota: i tassi sono in percentuale (es. 1.90 per 1.90%). Li convertiamo in decimali (es. 0.019).
    
    Restituisce:
      dates: lista ordinata di date
      discounts: array NumPy dei discount factors corrispondenti
    """
    # Converte la data di settlement in datetime.date
    today = to_date(datesSet.settle)
    
    # --- 1. Depos ---
    # Dividi per 100 per passare da "1.904" (1.904%) a 0.01904
    L_depos = ratesSet.depos["Mid"].values / 100.0
    
    # Converte le date dei Depos in datetime.date
    depos_dates = datesSet.depos["Settle Dates"].apply(to_date)
    
    # Frazione d'anno (ACT/360)
    y_frac_depos = np.array([yearfrac(today, d, mod.ACT_360) for d in depos_dates])
    
    # Calcolo discount factors Depos
    discounts_depos = 1.0 / (1.0 + y_frac_depos * L_depos)
    
    # --- 2. Futures ---
    futures = datesSet.future.copy()
    # Converte in date standard
    futures["Settle"] = futures["Settle"].apply(to_date)
    futures["Expiry"] = futures["Expiry"].apply(to_date)
    
    # Dividi i tassi Future per 100
    mid_rates_futures = ratesSet.future["Mid"].values / 100.0
    
    # Frazione d'anno (ACT/360)
    y_frac_futurs = futures.apply(
        lambda row: yearfrac(row["Settle"], row["Expiry"], mod.ACT_360), axis=1
    ).values
    
    # Calcolo dei fattori forward
    fwd_disc = 1.0 / (1.0 + y_frac_futurs * mid_rates_futures)
    
    discounts_futures = np.zeros(7)
    # Interpolazione su alcuni punti (MATLAB 1-based → Python 0-based)
    discounts_futures[0] = interpolation(
        depos_dates.iloc[2], depos_dates.iloc[3],
        discounts_depos[2], discounts_depos[3],
        futures.iloc[0]["Settle"], fwd_disc[0], today
    )
    discounts_futures[1] = interpolation(
        depos_dates.iloc[3], futures.iloc[0]["Expiry"],
        discounts_depos[3], discounts_futures[0],
        futures.iloc[1]["Settle"], fwd_disc[1], today
    )
    discounts_futures[2] = interpolation(
        futures.iloc[0]["Expiry"], futures.iloc[1]["Expiry"],
        discounts_futures[0], discounts_futures[1],
        futures.iloc[2]["Settle"], fwd_disc[2], today
    )
    discounts_futures[3] = discounts_futures[2] * fwd_disc[3]
    discounts_futures[4] = discounts_futures[3] * fwd_disc[4]
    discounts_futures[5] = interpolation(
        futures.iloc[3]["Expiry"], futures.iloc[4]["Expiry"],
        discounts_futures[3], discounts_futures[4],
        futures.iloc[5]["Settle"], fwd_disc[5], today
    )
    discounts_futures[6] = interpolation(
        futures.iloc[4]["Expiry"], futures.iloc[5]["Expiry"],
        discounts_futures[4], discounts_futures[5],
        futures.iloc[6]["Settle"], fwd_disc[6], today
    )
    
    # --- 3. Swaps ---
    # Genera date aggiuntive (50 anni) con add_Dates
    dt = date(2023, 2, 2)
    datesSet_add = add_Dates(dt, 50, mod_adjust.Modified)
    swap_dates = datesSet_add["Business Adjusted Dates"].apply(to_date).tolist()
    
    # Dividi i tassi Swap per 100
    mid_rates_swaps = ratesSet.swap["Mid"].values / 100.0
    
    # Frazione d'anno (EU_30_360) tra le date successive
    y_frac_swaps = np.array([
        yearfrac(swap_dates[i], swap_dates[i+1], mod.EU_30_360)
        for i in range(len(swap_dates) - 1)
    ])
    
    # Interpolazione spline sui nodi Swap
    swaps_numeric = np.array([to_date(x).toordinal() for x in datesSet.swap["Swap Dates"]])
    swap_dates_numeric = np.array([d.toordinal() for d in swap_dates])
    
    cs = CubicSpline(swaps_numeric, mid_rates_swaps)
    interpolated_rates = cs(swap_dates_numeric)
    
    discounts_swaps = np.zeros(len(swap_dates) - 1)
    # Interpolazione iniziale
    discounts_swaps[0] = interpolation(
        futures.iloc[2]["Expiry"], futures.iloc[3]["Expiry"],
        discounts_futures[2], discounts_futures[3],
        swap_dates[1], 1.0, today
    )
    
    # Calcolo iterativo
    for i in range(1, len(discounts_swaps)):
        x = np.dot(y_frac_swaps[:i], discounts_swaps[:i])
        discounts_swaps[i] = (1 - interpolated_rates[i+1] * x) / (1 + y_frac_swaps[i] * interpolated_rates[i+1])
    
    
    # --- 4. Aggregazione e ordinamento ---
    aggregated_dates = ([today] +
                        list(depos_dates.iloc[:4]) +
                        list(futures["Expiry"].iloc[:7]) +
                        swap_dates[2:])
    aggregated_discounts = np.concatenate((
        np.array([1.0]),
        discounts_depos[:4],
        discounts_futures,
        discounts_swaps[1:]
    ))
    
    combined = list(zip(aggregated_dates, aggregated_discounts))
    combined_sorted = sorted(combined, key=lambda x: x[0])
    dates_sorted, discounts_sorted = zip(*combined_sorted)

    # Trasforma separatamente i vettori in DataFrame pandas
    dates = pd.DataFrame({"Date": dates_sorted})
    discounts = pd.DataFrame({"Discount Factor": discounts_sorted})

    return dates, discounts
    


# Esempio d'uso
if __name__ == '__main__':
    # Esempio di come potresti creare e popolare datesSet e ratesSet
    class DatesSet:
        pass
    class RatesSet:
        pass

    # Date di settlement
    settlement_date = date(2023, 2, 3)

    # DataFrame di esempio per Depos, Futures, Swaps (usando date "fittizie")
    depos_dates = pd.DataFrame({
        "Settle Dates": [
            date(2023, 2, 3),
            date(2023, 2, 9),
            date(2023, 3, 2),
            date(2023, 4, 3)
        ]
    })
    futures_dates = pd.DataFrame({
        "Settle": [
            date(2023, 3, 15),
            date(2023, 6, 21),
            date(2023, 9, 20)
        ],
        "Expiry": [
            date(2023, 6, 15),
            date(2023, 9, 21),
            date(2023, 12, 20)
        ]
    })
    swaps_dates = pd.DataFrame({
        "Swap Dates": [
            date(2025, 2, 3),
            date(2026, 2, 2),
            date(2027, 2, 2)
        ]
    })

    # DataFrame di esempio per i tassi (in percentuale, quindi vanno divisi per 100)
    depos_rates = pd.DataFrame({
        "Bid": [1.9015, 1.89, 2.167, 2.507],
        "Ask": [1.9065, 1.9, 2.187, 2.517],
        "Mid": [1.904, 1.895, 2.177, 2.512]
    })
    futures_rates = pd.DataFrame({
        "Bid": [2.96, 3.445, 3.495],
        "Ask": [2.965, 3.45, 3.5],
        "Mid": [2.9625, 3.4475, 3.4975]
    })
    swaps_rates = pd.DataFrame({
        "Bid": [3.195, 3.021, 2.9135],
        "Ask": [3.2043, 3.029, 2.9226],
        "Mid": [3.19965, 3.025, 2.91805]
    })

    # Creazione oggetti con attributi
    datesSet = DatesSet()
    datesSet.settle = settlement_date
    datesSet.depos = depos_dates
    datesSet.future = futures_dates
    datesSet.swap = swaps_dates

    ratesSet = RatesSet()
    ratesSet.depos = depos_rates
    ratesSet.future = futures_rates
    ratesSet.swap = swaps_rates

    # Esecuzione del bootstrapping
    boot_dates, boot_discounts = bootstrap(datesSet, ratesSet)

    # Stampa dei risultati
    print("Aggregated (sorted) dates:")
    print(boot_dates)
    print("Aggregated (sorted) discounts:")
    print(boot_discounts)

    # Esempio di plot
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    plt.figure(figsize=(10, 6))
    plt.plot(boot_dates, boot_discounts, marker='o', linestyle='-', label='Discount Factor')
    plt.xlabel('Date')
    plt.ylabel('Discount Factor')
    plt.title('Bootstrapped Discount Curve')
    plt.grid(True)
    plt.legend()
    plt.gcf().autofmt_xdate()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.show()
