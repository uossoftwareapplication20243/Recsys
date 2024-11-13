import pandas as pd
import glob
import requests

# 모든 CSV 파일을 로드하여 데이터프레임으로 병합
file_paths = glob.glob('./matchv2_*.csv')


all_dfs = [pd.read_csv(file) for file in file_paths if pd.read_csv(file).shape[0] > 0]

#print(len(all_dfs))
# 파일이 비어있지 않을 경우에만 병합
df = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

# 중복된 행 제거
#df = df.drop_duplicates(subset=['ID', 'Win', 'NickName', 'Champion', 'item'])

if not df.empty:
    # 데이터 타입 변환 (Win 컬럼을 boolean으로 변환)
    df['Win'] = df['Win'].astype(bool)

    # 최신 챔피언 데이터 가져오기
    version_url = "https://ddragon.leagueoflegends.com/api/versions.json"
    versions = requests.get(version_url).json()
    latest_version = versions[0]  # 최신 버전

    champion_url = f"https://ddragon.leagueoflegends.com/cdn/{latest_version}/data/en_US/champion.json"
    response = requests.get(champion_url)
    champion_data = response.json()

    # 챔피언 이름 리스트 추출
    champion_names = [champion['name'] for champion in champion_data['data'].values()]

    # Champion 이름에서 'Strawberry_' 제거
    df['Champion'] = df['Champion'].str.replace('Strawberry_', '', regex=True)

    # ID별 챔피언 픽률 및 승률 계산
    id_champion_stats = df.groupby(['ID', 'Champion']).agg(
        GamesPlayed=('Champion', 'size'),
        Wins=('Win', lambda x: sum(x == True))
    ).reset_index()
    id_champion_stats['WinRate'] = (id_champion_stats['Wins'] / id_champion_stats['GamesPlayed']) * 100

    # 챔피언별 픽률 계산 (ID별 각 챔피언의 픽률 계산)
    id_total_games = df['ID'].value_counts().reset_index()
    id_total_games.columns = ['ID', 'TotalGames']
    id_champion_stats = pd.merge(id_champion_stats, id_total_games, on='ID')
    id_champion_stats['PickRate'] = (id_champion_stats['GamesPlayed'] / id_champion_stats['TotalGames']) * 100

    # 최신 챔피언만 필터링
    id_champion_stats = id_champion_stats[id_champion_stats['Champion'].isin(champion_names)]

    # CSV 파일로 저장
    id_champion_stats.to_csv('./id_champion_stats.csv', index=False)

    # 결과 출력
    unique_ids = df['ID'].nunique()
    print(unique_ids)
    print("ID별 챔피언 픽률 및 승률:")
    print(id_champion_stats)
else:
    print("병합할 데이터가 없습니다.")