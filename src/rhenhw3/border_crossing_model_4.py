#!/usr/bin/env python
# coding: utf-8


import sys
import argparse
from datetime import datetime
import math

import pandas as pd
from numpy.random import default_rng
import simpy
from pathlib import Path





class BorderCrossing(object):
    def __init__(self, env, num_officers, num_officers2,
                 mean_interarrival_time, pct_2nd_inspect,
                 primary_inspect_mean, primary_inspect_sd,
                 secondary_inspect_mean, secondary_inspect_sd,
                 rg
                 ):
        """
        Primary class that encapsulates border crossing resources.

        The detailed vehicle flow logic is in go_thru_customs() function.

        Parameters
        ----------
        env
        num_officers
        num_officers2
        mean_interarrival_time
        pct_2nd_inspect
        primary_inspect_mean
        primary_inspect_sd
        secondary_inspect_mean
        secondary_inspect_sd
        rg
        """

        # Simulation environment and random number generator
        self.env = env
        self.rg = rg

        # Create list to hold timestamps dictionaries (one per vehicle)
        self.timestamps_list = []
        # Create lists to hold occupancy tuples (time, occ)
        self.q_occupancy_list = [(0.0, 0.0)]
        self.insp1_occupancy_list = [(0.0, 0.0)]
        self.insp2_occupancy_list = [(0.0, 0.0)]

        # Create SimPy resources
        self.officer = simpy.Resource(env, num_officers)
        self.officer2 = simpy.Resource(env, num_officers2)

        # Initialize the vehicle flow related attributes
        self.mean_interarrival_time = mean_interarrival_time
        self.pct_2nd_inspect = pct_2nd_inspect

        self.primary_inspect_mean = primary_inspect_mean
        self.primary_inspect_sd = primary_inspect_sd
        self.secondary_inspect_mean = secondary_inspect_mean
        self.secondary_inspect_sd = secondary_inspect_sd

    # Create process methods
    def primary_inspect(self, vehicle):
        yield self.env.timeout(self.rg.normal(self.primary_inspect_mean, self.primary_inspect_sd))

    def secondary_inspect(self, vehicle):
        yield self.env.timeout(self.rg.normal(self.secondary_inspect_mean, self.secondary_inspect_sd))


def go_thru_customs(env, vehicle, crossing, quiet):
    """Defines the sequence of steps traversed by vehicles.

       Also capture a bunch of timestamps to make it easy to compute various system
       performance measures such as waiting times, queue sizes and resource utilization.
    """
    # Vehicle arrives at border crossing - note the arrival time
    arrival_ts = env.now

    # Update queue for 1st inspection - increment by 1
    q_prev_occ = crossing.q_occupancy_list[-1][1]
    q_new_occ = (env.now, q_prev_occ + 1)
    crossing.q_occupancy_list.append(q_new_occ)

    # Request customs officer for primary inspection
    with crossing.officer.request() as request:
        yield request
        got_officer_ts = env.now
        # We got one! Update inspection 1 occupancy - increment by 1
        prev_occ = crossing.insp1_occupancy_list[-1][1]
        new_occ = (env.now, prev_occ + 1)
        crossing.insp1_occupancy_list.append(new_occ)
        # and remove 1 occupant from the queue for 1st inspection
        crossing.q_occupancy_list.append((env.now, crossing.q_occupancy_list[-1][1] - 1))
        yield env.process(crossing.primary_inspect(vehicle))
        release_officer_ts = env.now
        crossing.insp1_occupancy_list.append((env.now, crossing.insp1_occupancy_list[-1][1] - 1))

    # Request officer for 2nd inspection if needed
    if crossing.rg.random() < crossing.pct_2nd_inspect:
        with crossing.officer2.request() as request:
            yield request
            got_officer2_ts = env.now
            # Update inspection 2 occupancy - increment by 1
            prev_occ2 = crossing.insp2_occupancy_list[-1][1]
            new_occ2 = (env.now, prev_occ2 + 1)
            crossing.insp2_occupancy_list.append(new_occ2)
            yield env.process(crossing.secondary_inspect(vehicle))
            release_officer2_ts = env.now
            crossing.insp2_occupancy_list.append((env.now, crossing.insp2_occupancy_list[-1][1] - 1))
    else:
        got_officer2_ts = pd.NA
        release_officer2_ts = pd.NA

    exit_system_ts = env.now
    if not quiet:
        print(f"Vehicle {vehicle} exited border crossing at time {env.now}")

    # Create dictionary of timestamps
    timestamps = {'vehicle_id': vehicle,
                  'arrival_ts': arrival_ts,
                  'got_officer_ts': got_officer_ts,
                  'release_officer_ts': release_officer_ts,
                  'got_officer2_ts': got_officer2_ts,
                  'release_officer2_ts': release_officer2_ts,
                  'exit_system_ts': exit_system_ts}

    crossing.timestamps_list.append(timestamps)


def run_crossing(env, crossing, stoptime=simpy.core.Infinity, max_arrivals=simpy.core.Infinity, quiet=False):
    """
    Run the border crossing for a specified amount of time or after generating a maximum number of vehicles.

    Parameters
    ----------
    env : SimPy environment
    crossing : ``Border Crossing`` object
    stoptime : float
    max_arrivals : int
    quiet : bool

    Yields
    -------
    Simpy environment timeout
    """

    # Create a counter to keep track of number of vehicles generated and to serve as unique vehicle id
    vehicle = 0

    # Loop for generating vehicles
    while env.now < stoptime and vehicle < max_arrivals:
        # Generate next interarrival time
        iat = crossing.rg.exponential(crossing.mean_interarrival_time)

        # This process will now yield to a 'timeout' event. This process will resume after iat time units.
        yield env.timeout(iat)

        # New vehicle generated = update counter of vehicles
        vehicle += 1

        if not quiet:
            print(f"Vehicle {vehicle} created at time {env.now}")

        # Register a go_thru_customs process for the new vehicle
        env.process(go_thru_customs(env, vehicle, crossing, quiet))

    print(f"{vehicle} vehicles processed.")


def compute_durations(timestamp_df):
    """Compute time durations of interest from timestamps dataframe and append new cols to dataframe"""

    timestamp_df['wait_for_1st_inspect'] = timestamp_df.loc[:, 'got_officer_ts'] - timestamp_df.loc[:, 'arrival_ts']
    timestamp_df['1st_inspect_time'] = timestamp_df.loc[:, 'release_officer_ts'] - timestamp_df.loc[:, 'got_officer_ts']
    timestamp_df['wait_for_2nd_inspect'] = timestamp_df.loc[:, 'got_officer2_ts'] - timestamp_df.loc[:, 'release_officer_ts']
    timestamp_df['2nd_inspect_time'] = timestamp_df.loc[:, 'release_officer2_ts'] - timestamp_df.loc[:, 'got_officer2_ts']
    timestamp_df['time_at_border'] = timestamp_df.loc[:, 'exit_system_ts'] - timestamp_df.loc[:, 'arrival_ts']

    return timestamp_df


def simulate(arg_dict, rep_num):
    """

    Parameters
    ----------
    arg_dict : dict whose keys are the input args
    rep_num : int, simulation replication number

    Returns
    -------
    Nothing returned but numerous output files written to ``args_dict[output_path]``

    """

    # Create a random number generator for this replication
    seed = arg_dict['seed'] + rep_num - 1
    rg = default_rng(seed=seed)

    # Resource capacity levels
    num_officers = arg_dict['num_officers']
    num_officers2 = arg_dict['num_officers2']

    # Initialize the vehicle flow related attributes
    vehicle_arrival_rate = arg_dict['vehicle_arrival_rate']
    mean_interarrival_time = 1.0 / (vehicle_arrival_rate / 60.0)

    pct_2nd_inspect = arg_dict['pct_2nd_inspect']
    primary_inspect_mean = arg_dict['primary_inspect_mean']
    primary_inspect_sd = arg_dict['primary_inspect_sd']
    secondary_inspect_mean = arg_dict['secondary_inspect_mean']
    secondary_inspect_sd = arg_dict['secondary_inspect_sd']

    # Other parameters
    stoptime = arg_dict['stoptime']  # No more arrivals after this time
    quiet = arg_dict['quiet']
    scenario = arg_dict['scenario']

    # Run the simulation
    env = simpy.Environment()

    # Create a crossing to simulate
    crossing = BorderCrossing(env, num_officers, num_officers2,
                              mean_interarrival_time, pct_2nd_inspect,
                              primary_inspect_mean, primary_inspect_sd,
                              secondary_inspect_mean, secondary_inspect_sd,
                              rg
                              )

    # Initialize and register the go_thru_customs generator function
    env.process(
        run_crossing(env, crossing, stoptime=stoptime, quiet=quiet))

    # Launch the simulation
    env.run()

    # Create output files and basic summary stats
    if len(arg_dict['output_path']) > 0:
        output_dir = Path.cwd() / arg_dict['output_path']
    else:
        output_dir = Path.cwd()

    # Create paths for the output logs
    crossing_vehicle_log_path = output_dir / f'crossing_vehicle_log_{scenario}_{rep_num}.csv'
    q_occupancy_df_path = output_dir / f'q_occupancy_{scenario}_{rep_num}.csv'
    insp1_occupancy_df_path = output_dir / f'insp1_occupancy_{scenario}_{rep_num}.csv'
    insp2_occupancy_df_path = output_dir / f'insp2_occupancy_{scenario}_{rep_num}.csv'

    # Create vehicle log dataframe and add scenario and rep number cols
    crossing_vehicle_log_df = pd.DataFrame(crossing.timestamps_list)
    crossing_vehicle_log_df['scenario'] = scenario
    crossing_vehicle_log_df['rep_num'] = rep_num

    # Reorder cols to get scenario and rep_num first
    num_cols = len(crossing_vehicle_log_df.columns)
    new_col_order = [-2, -1]
    new_col_order.extend([_ for _ in range(0, num_cols-2)])
    crossing_vehicle_log_df = crossing_vehicle_log_df.iloc[:, new_col_order]

    # Compute durations of interest for vehicle log
    crossing_vehicle_log_df = compute_durations(crossing_vehicle_log_df)

    # Create occupancy log dataframes and add scenario and rep number cols
    q_occupancy_df = pd.DataFrame(crossing.q_occupancy_list, columns=['ts', 'occ'])
    q_occupancy_df['scenario'] = scenario
    q_occupancy_df['rep_num'] = rep_num
    num_cols = len(q_occupancy_df.columns)
    new_col_order = [-2, -1]
    new_col_order.extend([_ for _ in range(0, num_cols - 2)])
    q_occupancy_df = q_occupancy_df.iloc[:, new_col_order]

    insp1_occupancy_df = pd.DataFrame(crossing.insp1_occupancy_list, columns=['ts', 'occ'])
    insp1_occupancy_df['scenario'] = scenario
    insp1_occupancy_df['rep_num'] = rep_num
    num_cols = len(insp1_occupancy_df.columns)
    new_col_order = [-2, -1]
    new_col_order.extend([_ for _ in range(0, num_cols - 2)])
    insp1_occupancy_df = insp1_occupancy_df.iloc[:, new_col_order]

    insp2_occupancy_df = pd.DataFrame(crossing.insp2_occupancy_list, columns=['ts', 'occ'])
    insp2_occupancy_df['scenario'] = scenario
    insp2_occupancy_df['rep_num'] = rep_num
    num_cols = len(insp2_occupancy_df.columns)
    new_col_order = [-2, -1]
    new_col_order.extend([_ for _ in range(0, num_cols - 2)])
    insp2_occupancy_df = insp2_occupancy_df.iloc[:, new_col_order]

    # Export logs to csv
    crossing_vehicle_log_df.to_csv(crossing_vehicle_log_path, index=False)
    q_occupancy_df.to_csv(q_occupancy_df_path, index=False)
    insp1_occupancy_df.to_csv(insp1_occupancy_df_path, index=False)
    insp2_occupancy_df.to_csv(insp2_occupancy_df_path, index=False)

    # Note simulation end time
    end_time = env.now
    print(f"Simulation replication {rep_num} ended at time {end_time}")


def process_sim_output(csvs_path, scenario):
    """

    Parameters
    ----------
    csvs_path : Path object for location of simulation output vehicle log csv files
    scenario : str

    Returns
    -------
    Dict of dicts

    Keys are:

    'vehicle_log_rep_stats' --> Contains dataframes from describe on group by rep num. Keys are perf measures.
    'vehicle_log_ci' -->        Contains dictionaries with overall stats and CIs. Keys are perf measures.
    """

    dest_path = csvs_path / f"consolidated_crossing_vehicle_log_{scenario}.csv"

    sort_keys = ['scenario', 'rep_num']

    # Create empty dict to hold the DataFrames created as we read each csv file
    dfs = {}

    # Loop over all the csv files
    for csv_f in csvs_path.glob('crossing_vehicle_log_*.csv'):
        # Split the filename off from csv extension. We'll use the filename
        # (without the extension) as the key in the dfs dict.
        fstem = csv_f.stem

        # Read the next csv file into a pandas DataFrame and add it to
        # the dfs dict.
        df = pd.read_csv(csv_f)
        dfs[fstem] = df

    # Use pandas concat method to combine the file specific DataFrames into
    # one big DataFrame.
    vehicle_log_df = pd.concat(dfs)

    # Since we didn't try to control the order in which the files were read,
    # we'll sort the final DataFrame in place by the specified sort keys.
    vehicle_log_df.sort_values(sort_keys, inplace=True)

    # Export the final DataFrame to a csv file. Suppress the pandas index.
    vehicle_log_df.to_csv(dest_path, index=False)

    # Compute summary statistics for several performance measures
    vehicle_log_stats = summarize_vehicle_log(vehicle_log_df, scenario)

    # Now delete the individual replication files
    for csv_f in csvs_path.glob('crossing_vehicle_log_*.csv'):
        csv_f.unlink()

    return vehicle_log_stats


def summarize_vehicle_log(vehicle_log_df, scenario):
    """

    Parameters
    ----------
    vehicle_log_df : DataFrame created by process_sim_output
    scenario : str

    Returns
    -------
    Dict of dictionaries - See comments below
    """

    # Create empty dictionaries to hold computed results
    vehicle_log_rep_stats = {}  # Will store dataframes from describe on group by rep num. Keys are perf measures.
    vehicle_log_ci = {}         # Will store dictionaries with overall stats and CIs. Keys are perf measures.
    vehicle_log_stats = {}      # Container dict returned by this function containing the two previous dicts.

    # Create list of performance measures for looping over
    performance_measures = ['wait_for_1st_inspect', '1st_inspect_time', 'wait_for_2nd_inspect',
                            '2nd_inspect_time', 'time_at_border']

    for pm in performance_measures:
        # Compute descriptive stats for each replication and store dataframe in dict
        vehicle_log_rep_stats[pm] = vehicle_log_df.groupby(['rep_num'])[pm].describe()
        # Compute across replication stats
        n_samples = vehicle_log_rep_stats[pm]['mean'].count()
        mean_mean = vehicle_log_rep_stats[pm]['mean'].mean()
        sd_mean = vehicle_log_rep_stats[pm]['mean'].std()
        ci_95_lower = mean_mean - 1.96 * sd_mean / math.sqrt(n_samples)
        ci_95_upper = mean_mean + 1.96 * sd_mean / math.sqrt(n_samples)
        # Store cross replication stats as dict in dict
        vehicle_log_ci[pm] = {'n_samples': n_samples, 'mean_mean': mean_mean, 'sd_mean': sd_mean,
                              'ci_95_lower': ci_95_lower, 'ci_95_upper': ci_95_upper}

    vehicle_log_stats['scenario'] = scenario
    vehicle_log_stats['vehicle_log_rep_stats'] = vehicle_log_rep_stats
    # Convert the final summary stats dict to a DataFrame
    vehicle_log_stats['vehicle_log_ci'] = pd.DataFrame(vehicle_log_ci)

    return vehicle_log_stats


def process_command_line():
    """
    Parse command line arguments

    `argv` is a list of arguments, or `None` for ``sys.argv[1:]``.
    Return a Namespace representing the argument list.
    """

    # Create the parser
    parser = argparse.ArgumentParser(prog='border_crossing_model_4',
                                     description='Run border crossing simulation')

    # Add arguments
    parser.add_argument(
        "--config", type=str, default=None,
        help="Configuration file containing input parameter arguments and values"
    )

    parser.add_argument("--vehicle_arrival_rate", default=60, help="vehicles per hour",
                        type=float)

    parser.add_argument("--num_officers", default=3, help="number of 1st inspection officers",
                        type=int)

    parser.add_argument("--num_officers2", default=2, help="number of 2nd inspection officers",
                        type=int)

    parser.add_argument("--pct_2nd_inspect", default=0.1,
                        help="percent of vehicles pulled aside for 2nd inspection (default = 0.1)",
                        type=float)

    parser.add_argument("--primary_inspect_mean", default=5.0,
                        help="Mean time (units) for primary inspection (default = 5.0)",
                        type=float)

    parser.add_argument("--primary_inspect_sd", default=1,
                        help="Standard deviation time (units) for primary inspection (default = 1)",
                        type=float)

    parser.add_argument("--secondary_inspect_mean", default=15.0,
                        help="Mean time (units) for secondary inspection (default = 15.0)",
                        type=float)

    parser.add_argument("--secondary_inspect_sd", default=2.5,
                        help="Standard deviation time (units) for secondary inspection (default = 2.5)",
                        type=float)

    parser.add_argument(
        "--scenario", type=str, default=datetime.now().strftime("%Y.%m.%d.%H.%M."),
        help="Appended to output filenames."
    )

    parser.add_argument("--stoptime", default=480, help="time that simulation stops (default = 480)",
                        type=float)

    parser.add_argument("--num_reps", default=1, help="number of simulation replications (default = 1)",
                        type=int)

    parser.add_argument("--seed", default=3, help="random number generator seed (default = 3)",
                        type=int)

    parser.add_argument(
        "--output_path", type=str, default="output", help="location for output file writing")

    parser.add_argument("--quiet", action='store_true',
                        help="If True, suppresses output messages (default=False")

    # do the parsing
    args = parser.parse_args()

    if args.config is not None:
        # Read inputs from config file
        with open(args.config, "r") as fin:
            args = parser.parse_args(fin.read().split())

    return args


def main():

    args = process_command_line()
    print(args)

    num_reps = args.num_reps
    scenario = args.scenario

    if len(args.output_path) > 0:
        output_dir = Path.cwd() / args.output_path
    else:
        output_dir = Path.cwd()

    # Main simulation replication loop
    for i in range(1, num_reps + 1):
        simulate(vars(args), i)

    # Consolidate the vehicle logs and compute summary stats
    vehicle_log_stats = process_sim_output(output_dir, scenario)
    print(f"\nScenario: {scenario}")
    pd.set_option("display.precision", 3)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 120)
    print(vehicle_log_stats['vehicle_log_rep_stats'])
    print(vehicle_log_stats['vehicle_log_ci'])


if __name__ == '__main__':
    main()


