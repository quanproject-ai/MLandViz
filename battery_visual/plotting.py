import pandas as pd
import plot_helper as ph

SINGLETESTFILEPATH = (
    "G:\\All Coding Project\\machinelearning\\dataset\\timeseriescell\\"
)
METADATAFILEPATH = "G:\\All Coding Project\\machinelearning\\dataset\\celldata.csv"
BATTERY_GROUP = {
    "1": ["B0025", "B0026", "B0027", "B0028"],
    "2b": ["B0029", "B0030", "B0031", "B0032"],
    "2c": ["B0033", "B0034", "B0036"],
    "2d": ["B0038", "B0039", "B0040"],
    "2e": ["B0041", "B0042", "B0043", "B0044"],
    "3": ["B0045", "B0046", "B0047", "B0048"],
    "4": ["B0049", "B0050", "B0051", "B0052"],
    "5": ["B0053", "B0054", "B0055", "B0056"],
    "6": ["B0005", "B0006", "B0007", "B0018"],
}
# https://www.kaggle.com/code/susanketsarkar/rul-prediction-using-variational-autoencoder-lstm
# Create box plots for each battery in each group
# Create capacity vs cycle
metadata_df = pd.read_csv(filepath_or_buffer=METADATAFILEPATH)


def plot_capacity_boxplot(group_key: str):
    color_mod = 0
    fig = ph._generate_subplot(
        rows=1,
        cols=1,
        legend_title="battery id",
        x_title="battery id",
        y_title="discharge capacity [Ah]",
    )
    fig.update_layout(title=f"Discharge Capacity vs. Battery ID Group #{group_key}")
    for value in BATTERY_GROUP[group_key]:
        df_to_plot = metadata_df[
            (metadata_df["battery_id"] == value) & (metadata_df["type"] == "discharge")
        ]
        ph._boxplot(
            fig,
            y=df_to_plot["Capacity"].astype(float),  # data sanitizing#
            x=value,
            color=ph.COLORS[color_mod % len(ph.COLORS)],
            row=1,
            col=1,
        )
        color_mod += 1
    fig.show()
    return fig


def plot_capacity_vs_cycle_scatter(group_key: str):
    fig = ph._generate_subplot(
        rows=1,
        cols=1,
        legend_title="battery id",
        x_title="cycle numbers",
        y_title="discharge capacity [Ah]",
    )
    fig.update_layout(
        title=f"Discharge Capacity vs. Cycle numbers for Battery ID Group #{group_key}"
    )
    color_mod = 0
    for value in BATTERY_GROUP[group_key]:
        df_to_plot = metadata_df[
            (metadata_df["battery_id"] == value) & (metadata_df["type"] == "discharge")
        ].copy()
        df_to_plot["cycle_number"] = range(len(df_to_plot))
        ph._scatter(
            fig,
            name=value,
            y=df_to_plot["Capacity"].astype(float).ffill(),  # data sanitizing#
            x=df_to_plot["cycle_number"],
            colors=ph.COLORS[color_mod % len(ph.COLORS)],
            row=1,
            col=1,
        )
        color_mod += 1
    return fig


def plot_discharge_parameters_over_time(filepath: str, group_key: str):
    color_mod = 0
    fig = ph._generate_subplot(
        rows=3,
        cols=2,
        legend_title="Discharge Paremeters vs. Time",
        x_title="",
        y_title="",
    )
    ##can be re-written for better logic than copy and paste#
    for rows in range(1, 4):
        for cols in range(1, 3):
            fig.update_xaxes(title_text="Time [s]", row=rows, col=cols)
    fig.update_yaxes(title_text="Voltage_measured [V]", row=1, col=1)
    fig.update_yaxes(title_text="Current_measured [A]", row=2, col=1)
    fig.update_yaxes(title_text="Temperature_measured [C]", row=3, col=1)
    fig.update_yaxes(title_text="Voltage_load [V]", row=1, col=2)
    fig.update_yaxes(title_text="Current_load [A]", row=2, col=2)
    for value in BATTERY_GROUP[group_key]:
        metadf_filter = metadata_df[(metadata_df['type'] == 'discharge') & (metadata_df['battery_id']==value)].copy()
        plot_list = (metadf_filter['filename'].tolist())
        for name in plot_list:
            fullfilepath = filepath + name
            df = pd.read_csv(filepath_or_buffer=fullfilepath)
            ph._scatter(
                fig=fig,
                name=name,
                x=df["Time"],
                y=df["Voltage_measured"],
                colors=ph.COLORS[color_mod % len(ph.COLORS)],
                row=1,
                col=1,
            )
            ph._scatter(
                fig=fig,
                name=name,
                x=df["Time"],
                y=df["Current_measured"],
                colors=ph.COLORS[color_mod % len(ph.COLORS)],
                row=2,
                col=1,
                legend = False
            )
            ph._scatter(
                fig=fig,
                name=name,
                x=df["Time"],
                y=df["Temperature_measured"],
                colors=ph.COLORS[color_mod % len(ph.COLORS)],
                row=3,
                col=1,
                legend = False
            )
            ph._scatter(
                fig=fig,
                name=name,
                x=df["Time"],
                y=df["Voltage_load"],
                colors=ph.COLORS[color_mod % len(ph.COLORS)],
                row=1,
                col=2,
                legend = False
            )
            ph._scatter(
                fig=fig,
                name=name,
                x=df["Time"],
                y=df["Current_load"],
                colors=ph.COLORS[color_mod % len(ph.COLORS)],
                row=2,
                col=2,
                legend = False
            )
            color_mod +=1
    fig.show()
    return fig





