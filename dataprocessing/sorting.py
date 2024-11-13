import pandas as pd
import matplotlib.pyplot as plt

def calculate_pickrate_statistics(file_path):
    # Load the CSV file
    champion_stats = pd.read_csv(file_path)

    # Calculate basic statistics for PickRate
    pickrate_mean = champion_stats['PickRate'].mean()
    pickrate_median = champion_stats['PickRate'].median()
    pickrate_std = champion_stats['PickRate'].std()
    pickrate_min = champion_stats['PickRate'].min()
    pickrate_max = champion_stats['PickRate'].max()

    # Print the statistics
    print(f"PickRate Statistics:")
    print(f"Mean: {pickrate_mean}")
    print(f"Median: {pickrate_median}")
    print(f"Standard Deviation: {pickrate_std}")
    print(f"Min: {pickrate_min}")
    print(f"Max: {pickrate_max}")

def plot_pickrate_distribution(file_path):
    # Load the CSV file
    champion_stats = pd.read_csv(file_path)

    # Filter PickRate to be between 0 and 40
    filtered_stats = champion_stats[champion_stats['PickRate'].between(0, 40)]

    # Plot the distribution of PickRate within 0% to 40%
    plt.figure(figsize=(10, 6))
    plt.hist(filtered_stats['PickRate'], bins=30, edgecolor='black')
    plt.title('Distribution of PickRate (0% to 40%)')
    plt.xlabel('PickRate')
    plt.ylabel('Frequency')
    plt.grid(True)

    # Show the plot
    plt.show()

def generate_user_preference_stats(file_path):
    # Load the CSV file
    champion_stats = pd.read_csv(file_path)

    # Filter champions with more than 10 games played
    filtered_stats = champion_stats[champion_stats['GamesPlayed'] >= 10]

    # Filter by WinRate > 50 and sort by WinRate for each user
    user_champion_stats_winrate = filtered_stats[filtered_stats['WinRate'] > 50]
    user_champion_stats_winrate = user_champion_stats_winrate.sort_values(by=['ID', 'WinRate'], ascending=[True, False])
    user_champion_grouped_winrate = user_champion_stats_winrate.groupby('ID')['Champion'].apply(list).reset_index()
    user_champion_grouped_winrate.rename(columns={'Champion': 'WinRate_Sorted_Champions'}, inplace=True)

    # Filter by PickRate > 10 and sort by PickRate for each user
    user_champion_stats_pickrate = filtered_stats[filtered_stats['PickRate'] > 7.5]
    user_champion_stats_pickrate = user_champion_stats_pickrate.sort_values(by=['ID', 'PickRate'], ascending=[True, False])
    user_champion_grouped_pickrate = user_champion_stats_pickrate.groupby('ID')['Champion'].apply(list).reset_index()
    user_champion_grouped_pickrate.rename(columns={'Champion': 'PickRate_Sorted_Champions'}, inplace=True)

    # Merge WinRate and PickRate data
    user_champion_grouped = pd.merge(user_champion_grouped_winrate, user_champion_grouped_pickrate, on='ID')

    # Save the user-champion preferences to csv file
    user_champion_grouped.to_csv('./user_winrate_pickrate_sorted_champions.csv', index=False)
import pandas as pd

def calculate_significant_pickrate_threshold(file_path):
    # Load the CSV file
    champion_stats = pd.read_csv(file_path)

    # Calculate median, 75th percentile, and mean + standard deviation
    pickrate_median = champion_stats['PickRate'].median()
    pickrate_75th_percentile = champion_stats['PickRate'].quantile(0.75)
    pickrate_mean = champion_stats['PickRate'].mean()
    pickrate_std = champion_stats['PickRate'].std()
    pickrate_mean_plus_std = pickrate_mean + pickrate_std

    # Print the thresholds
    print(f"Median PickRate: {pickrate_median}")
    print(f"75th Percentile PickRate: {pickrate_75th_percentile}")
    print(f"Mean + 1 Std Dev PickRate: {pickrate_mean_plus_std}")


def calculate_relative_pickrate(file_path):
    # Load the CSV file
    champion_stats = pd.read_csv(file_path)

    # Group by User ID and calculate the total games played per user
    user_total_games = champion_stats.groupby('ID')['GamesPlayed'].sum().reset_index()
    user_total_games.rename(columns={'GamesPlayed': 'TotalGamesPlayed'}, inplace=True)

    # Merge the total games back into the champion_stats dataframe
    champion_stats = pd.merge(champion_stats, user_total_games, on='ID')

    # Calculate each champion's pickrate as a percentage of the user's total games
    champion_stats['RelativePickRate'] = (champion_stats['GamesPlayed'] / champion_stats['TotalGamesPlayed']) * 100

    # Print statistics to see significant pickrates (Median, 75th Percentile, etc.)
    relative_pickrate_median = champion_stats['RelativePickRate'].median()
    relative_pickrate_75th_percentile = champion_stats['RelativePickRate'].quantile(0.75)
    relative_pickrate_mean = champion_stats['RelativePickRate'].mean()
    relative_pickrate_std = champion_stats['RelativePickRate'].std()
    relative_pickrate_mean_plus_std = relative_pickrate_mean + relative_pickrate_std

    print(f"Median Relative PickRate: {relative_pickrate_median}")
    print(f"75th Percentile Relative PickRate: {relative_pickrate_75th_percentile}")
    print(f"Mean + 1 Std Dev Relative PickRate: {relative_pickrate_mean_plus_std}")

    # Return the dataframe with the new RelativePickRate column for further analysis
    return champion_stats

file_path = './id_champion_stats.csv'  # CSV 파일 경로를 여기에 넣으세요
#relative_pickrate_df = calculate_relative_pickrate(file_path)
#calculate_significant_pickrate_threshold(file_path)
#calculate_pickrate_statistics(file_path)
#plot_pickrate_distribution(file_path)
generate_user_preference_stats(file_path)