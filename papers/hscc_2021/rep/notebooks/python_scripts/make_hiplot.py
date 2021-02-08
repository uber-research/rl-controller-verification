# Observations on the aggregate dataframe
# Needs to install hiplot first

import pandas as pd
import hiplot as hip

def create_hiplot():
    """Create mean_aggregated_all.csv and mean_by_checkpoint.html
    
    It merges two dataframes, indeed, when the ri
    """
    # Put the right path
    path = "aggregated_all.csv"
    data = pd.read_csv(path) 
    data_bis = data.loc[data["rising_time_all count"]>0]
    groub_by_columns = ["nb_neurons", "nb_layers", "algo", "pid_rates", "thrust", "p", "nof_training_iterations", "test_windgust_magnitude_max", "training_windgust_magnitude_max", "training_saturation_motor", "test_saturation_motor"]
    data["OK rising t."] = data["rising_time_all count"] / (data["rising_time_bad count"] + data["rising_time_all count"]) 
    data["OK off."] = 1 - data["offset_bad count"] / data["offset_all count"]
    data["OK overshoot"] = 1 - data["overshoot_bad count"] / data["overshoot_all count"]
    data_mean = data.groupby(groub_by_columns).mean()
    data_mean.drop(["rising_time_all avg", "rising_time_all max"], axis=1, inplace=True)
    data_mean_bis = data_bis.groupby(groub_by_columns).mean()[["rising_time_all avg", "rising_time_all max"]]
    data_final = data_mean.merge(data_mean_bis, how="left", on=groub_by_columns).reset_index()
    # Put the column names you want to observe
    selected_columns = ["nb_neurons", "nb_layers", "algo", "pid_rates", "thrust", "p", "training_windgust_magnitude_max", "test_windgust_magnitude_max", "training_saturation_motor", "test_saturation_motor", "nof_training_iterations", "OK rising t.", "OK off.", "OK overshoot", "avg rising t.", "avg off.", "avg overshoot", "max rising t.", "max off.", "max overshoot"]
    renamed_columns_offset = {"offset_all_rel avg":"avg off.", "offset_all_rel max":"max off."}
    renamed_columns_overshoot = {"overshoot_all_rel avg":"avg overshoot", "overshoot_all_rel max":"max overshoot"}
    renamed_columns_rising_time = {"rising_time_all avg":"avg rising t.", "rising_time_all max":"max rising t."}
    renamed_columns = [renamed_columns_offset, renamed_columns_overshoot, renamed_columns_rising_time]
    for col in renamed_columns:
        data_final = data_final.rename(columns=col)
    data_mean_all = data_final[selected_columns]
    path_mean = 'mean_by_checkpoint.html'

    # Saves the interactive graph
    data_mean_all.to_csv("mean_aggregated_all.csv")
    hip_exp = hip.Experiment.from_dataframe(data_mean_all)
    hip_exp.to_html(path_mean)
    hip_exp.display()
