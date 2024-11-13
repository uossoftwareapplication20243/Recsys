
import requests
import csv
import time
API_KEY = 'RGAPI-ff474177-b834-4d37-899d-1127454a7f5f'  # Riot API 키를 입력하세요.
REGION = 'kr'  # 한국 서버
URL = f'https://{REGION}.api.riotgames.com/lol/league/v4/challengerleagues/by-queue/RANKED_SOLO_5x5'
headers = {
    'X-Riot-Token': API_KEY
}
def league_v4_queue_tier_division(queue, tier, division, page_number):
    if division == 1 or tier in ["CHALLENGER", "GRANDMASTER", "MASTER"]:
        division = 'I'
    elif division == 2 :
        division = 'II'
    elif division == 3 :
        division = 'III'
    elif division == 4:
        division = 'IV'
    if queue == "solo" :
        queue = "RANKED_SOLO_5x5"
    elif queue == "free" :
        queue = "RANKED_FLEX_SR"
    url = f"https://kr.api.riotgames.com/lol/league-exp/v4/entries/{queue}/{tier}/{division}?page={page_number}"
    print(url)
    return requests.get(url, headers=headers)


with open('../high_tier_user.csv', mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(
        ['Summoner ID', 'League Points', 'Rank', 'Wins', 'Losses', 'Veteran', 'Inactive',
         'FreshBlood',
         'HotStreak'])

    for tier in ["CHALLENGER", "GRANDMASTER", "MASTER","DIAMOND"]:
        if tier in ["CHALLENGER", "GRANDMASTER", "MASTER"]:
            range1 =1
        else:
            range1 =4

        for j in range(range1):
            for i in range(40):
                time.sleep(1)
                response = league_v4_queue_tier_division("solo", tier, j + 1, i + 1)

                if response.status_code == 200:
                    data = response.json()
                    # 데이터 작성
                    for entry in data:
                        writer.writerow([
                            entry.get('summonerId', 'Unknown'),
                            entry.get('leaguePoints', 0),
                            entry.get('rank', 'Unknown'),
                            entry.get('wins', 0),
                            entry.get('losses', 0),
                            entry.get('veteran', False),
                            entry.get('inactive', False),
                            entry.get('freshBlood', False),
                            entry.get('hotStreak', False)
                        ])

                else:
                    print(f"Error: API 요청 실패 - 상태 코드 {response.status_code}")





print("CSV 파일 저장 완료")